"""Calibrated control-plane demo.

This example does not load a language model. It shows the product-facing part of
EigenTruth's control workflow: load or create a calibration artifact, evaluate
diagnostics with `RiskController`, verify simple claims, execute actions through
`ActionExecutorRegistry`, and emit a JSON `ProductTrace`.

The output is a trace for routing and debugging. It is not proof that a response
is true. When the repository's SmolLM2 l80 calibration artifact is present, it is
used by default; otherwise the script falls back to the Qwen l80 artifact and
then toy thresholds. When the SmolLM2 adapter-gated release candidate is
present, its product promotion contract supplies the default verifier route and
adapter-family metadata; pass it explicitly with `--promotion-contract` to also
enforce its runtime budget policy.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from eigentruth.adapters import CachedRetriever, CalculatorVerifier, InMemoryRetriever, RetrievalActionExecutor
from eigentruth.calibration import CalibrationArtifact, CalibrationScore
from eigentruth.control import (
    RUNTIME_PROFILE_NAMES,
    ActionExecutorRegistry,
    ControlAction,
    ProductPromotionContract,
    ProductRuntimeBudgetPolicy,
    RiskController,
    RuntimeProfile,
    StagedVerificationPolicy,
    evaluate_product_runtime_budget,
    get_runtime_profile,
    run_verification_loop,
)
from eigentruth.registry import ArtifactRegistry
from eigentruth.verify import (
    CachedVerifier,
    GroundednessVerifier,
    InMemoryVerifier,
    RoutedVerifier,
    VerificationStatus,
    Verifier,
    VerifierRoute,
    extract_claims,
    normalize_claim_text,
)

DEFAULT_TEXT = "Paris is the capital of France. The moon is made of cheese."
DEFAULT_SMOLLM2_ARTIFACT_PATH = (
    Path(__file__).resolve().parents[1] / "artifacts" / "smollm2_truthfulqa_l80_best_calibration.json"
)
DEFAULT_QWEN_ARTIFACT_PATH = (
    Path(__file__).resolve().parents[1] / "artifacts" / "qwen05_truthfulqa_l80_best_calibration.json"
)
DEFAULT_PROMOTION_CONTRACT_PATH = (
    Path(__file__).resolve().parents[1]
    / "artifacts"
    / "smollm2_l20_inside_trigger_budget_derived_adapter_gated_staged_release_candidate_registry_workflow.json"
)
ARITHMETIC_TEXT_PATTERN = r"\d[\d\s().%+*/-]*[+*/%-][\d\s().%+*/-]*(?:=|equals|is)\s*[-+]?\d"


def toy_artifact() -> CalibrationArtifact:
    """Return a tiny artifact for running the demo without benchmark outputs."""
    return CalibrationArtifact(
        model_id="demo-model",
        target_layer=-1,
        scores=(
            CalibrationScore("maha_last", threshold=3.0, conformal_alpha=0.2),
            CalibrationScore("subspace_resid", threshold=1.2, conformal_alpha=0.2),
        ),
        eigentruth_version="0.1.0",
        calibration_dataset_metadata={
            "source": "examples/calibrated_control_demo.py",
            "note": "toy thresholds for demonstration only",
        },
    )


def default_artifact_path() -> Path | None:
    """Return the preferred repository calibration artifact when available."""
    for path in (DEFAULT_SMOLLM2_ARTIFACT_PATH, DEFAULT_QWEN_ARTIFACT_PATH):
        if path.exists():
            return path
    return None


def default_promotion_contract_path() -> Path | None:
    """Return the preferred product promotion contract when available."""
    return DEFAULT_PROMOTION_CONTRACT_PATH if DEFAULT_PROMOTION_CONTRACT_PATH.exists() else None


def default_artifact() -> CalibrationArtifact:
    """Return the preferred demo artifact, falling back to toy thresholds."""
    path = default_artifact_path()
    if path is not None:
        return CalibrationArtifact.load_json(path)
    return toy_artifact()


def load_artifact(path: str | None) -> CalibrationArtifact:
    """Load a calibration artifact or return the built-in demo artifact."""
    if path is None:
        return default_artifact()
    return CalibrationArtifact.load_json(path)


def artifact_source(path: str | None) -> str:
    """Return a stable source label for trace metadata."""
    if path is not None:
        return str(Path(path))
    default_path = default_artifact_path()
    if default_path is not None:
        return str(default_path.relative_to(Path(__file__).resolve().parents[1]))
    return "builtin-toy-artifact"


def default_diagnostics_for_artifact(artifact: CalibrationArtifact) -> dict[str, float]:
    """Return diagnostics that cross each finite artifact threshold."""
    diagnostics = {}
    for score in artifact.scores:
        if not math.isfinite(score.threshold):
            continue
        margin = max(abs(score.threshold) * 0.10, 1e-3)
        if score.direction == "higher":
            diagnostics[score.name] = float(score.threshold + margin)
        else:
            diagnostics[score.name] = float(score.threshold - margin)
    return diagnostics


def parse_json_mapping(value: str, *, name: str) -> dict[str, Any]:
    """Parse a JSON object from a CLI argument."""
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError(f"{name} must be a JSON object.")
    return parsed


def parse_json_sequence(value: str, *, name: str) -> list[Any]:
    """Parse a JSON list from a CLI argument."""
    parsed = json.loads(value)
    if not isinstance(parsed, list):
        raise ValueError(f"{name} must be a JSON list.")
    return parsed


def runtime_budget_policy_from_args(args: argparse.Namespace) -> ProductRuntimeBudgetPolicy | None:
    """Build an optional runtime budget policy from CLI-like arguments."""
    promotion_contract_path = getattr(args, "promotion_contract", None)
    promotion_contract = getattr(args, "_promotion_contract", None)
    base_policy = (
        None
        if promotion_contract_path is None
        else (
            promotion_contract.runtime_budget_policy
            if isinstance(promotion_contract, ProductPromotionContract)
            else ProductPromotionContract.from_json(promotion_contract_path).runtime_budget_policy
        )
    )
    max_total_seconds = getattr(args, "max_runtime_total_seconds", None)
    raw_phase_seconds = getattr(args, "max_runtime_phase_seconds", None)
    raw_phase_p95_seconds = getattr(args, "max_runtime_phase_p95_seconds", None)
    raw_phase_p99_seconds = getattr(args, "max_runtime_phase_p99_seconds", None)
    max_mean_route_duration_seconds = getattr(args, "max_mean_route_duration_seconds", None)
    max_p95_route_duration_seconds = getattr(args, "max_p95_route_duration_seconds", None)
    max_p99_route_duration_seconds = getattr(args, "max_p99_route_duration_seconds", None)
    max_route_duration_seconds = getattr(args, "max_route_duration_seconds", None)
    max_mean_attempted_route_count = getattr(args, "max_mean_attempted_route_count", None)
    max_retrieval_use_rate = getattr(args, "max_retrieval_use_rate", None)
    max_retrieval_hit_count = getattr(args, "max_retrieval_hit_count", None)
    min_cache_hit_rate = getattr(args, "min_cache_hit_rate", None)
    raw_named_cache_hit_rate = getattr(args, "min_named_cache_hit_rate", None)
    if (
        base_policy is None
        and max_total_seconds is None
        and raw_phase_seconds is None
        and raw_phase_p95_seconds is None
        and raw_phase_p99_seconds is None
        and max_mean_route_duration_seconds is None
        and max_p95_route_duration_seconds is None
        and max_p99_route_duration_seconds is None
        and max_route_duration_seconds is None
        and max_mean_attempted_route_count is None
        and max_retrieval_use_rate is None
        and max_retrieval_hit_count is None
        and min_cache_hit_rate is None
        and raw_named_cache_hit_rate is None
    ):
        return None
    base_policy = base_policy or ProductRuntimeBudgetPolicy()
    phase_seconds = (
        dict(base_policy.max_phase_seconds)
        if raw_phase_seconds is None
        else parse_json_mapping(raw_phase_seconds, name="--max-runtime-phase-seconds")
    )
    phase_p95_seconds = (
        dict(base_policy.max_phase_p95_seconds)
        if raw_phase_p95_seconds is None
        else parse_json_mapping(raw_phase_p95_seconds, name="--max-runtime-phase-p95-seconds")
    )
    phase_p99_seconds = (
        dict(base_policy.max_phase_p99_seconds)
        if raw_phase_p99_seconds is None
        else parse_json_mapping(raw_phase_p99_seconds, name="--max-runtime-phase-p99-seconds")
    )
    named_cache_hit_rate = (
        dict(base_policy.min_named_cache_hit_rate)
        if raw_named_cache_hit_rate is None
        else parse_json_mapping(raw_named_cache_hit_rate, name="--min-named-cache-hit-rate")
    )
    return ProductRuntimeBudgetPolicy(
        max_total_seconds=base_policy.max_total_seconds if max_total_seconds is None else max_total_seconds,
        max_phase_seconds={key: float(value) for key, value in phase_seconds.items()},
        max_phase_p95_seconds={key: float(value) for key, value in phase_p95_seconds.items()},
        max_phase_p99_seconds={key: float(value) for key, value in phase_p99_seconds.items()},
        max_mean_route_duration_seconds=(
            base_policy.max_mean_route_duration_seconds
            if max_mean_route_duration_seconds is None
            else max_mean_route_duration_seconds
        ),
        max_p95_route_duration_seconds=(
            base_policy.max_p95_route_duration_seconds
            if max_p95_route_duration_seconds is None
            else max_p95_route_duration_seconds
        ),
        max_p99_route_duration_seconds=(
            base_policy.max_p99_route_duration_seconds
            if max_p99_route_duration_seconds is None
            else max_p99_route_duration_seconds
        ),
        max_route_duration_seconds=(
            base_policy.max_route_duration_seconds
            if max_route_duration_seconds is None
            else max_route_duration_seconds
        ),
        max_mean_attempted_route_count=(
            base_policy.max_mean_attempted_route_count
            if max_mean_attempted_route_count is None
            else max_mean_attempted_route_count
        ),
        max_retrieval_use_rate=(
            base_policy.max_retrieval_use_rate
            if max_retrieval_use_rate is None
            else max_retrieval_use_rate
        ),
        max_retrieval_hit_count=(
            base_policy.max_retrieval_hit_count
            if max_retrieval_hit_count is None
            else max_retrieval_hit_count
        ),
        min_cache_hit_rate=base_policy.min_cache_hit_rate if min_cache_hit_rate is None else min_cache_hit_rate,
        min_named_cache_hit_rate={key: float(value) for key, value in named_cache_hit_rate.items()},
    )


def low_diagnostics_for_artifact(artifact: CalibrationArtifact) -> dict[str, float]:
    """Return diagnostics that stay below each finite artifact threshold."""
    diagnostics = {}
    for score in artifact.scores:
        if not math.isfinite(score.threshold):
            continue
        margin = max(abs(score.threshold) * 0.10, 1e-3)
        if score.direction == "higher":
            diagnostics[score.name] = float(score.threshold - margin)
        else:
            diagnostics[score.name] = float(score.threshold + margin)
    return diagnostics


def runtime_profile_metadata(profile: RuntimeProfile | None) -> dict[str, Any]:
    """Return trace metadata for the selected runtime profile."""
    if profile is None:
        return {
            "runtime_profile": None,
            "runtime_profile_control_defaults": None,
        }
    return {
        "runtime_profile": profile.name,
        "runtime_profile_description": profile.description,
        "runtime_profile_control_defaults": dict(profile.control_defaults),
    }


def stage_policy_from_runtime_profile(
    profile: RuntimeProfile | None,
    *,
    staged_verification: bool | None = None,
) -> StagedVerificationPolicy | None:
    """Build a staged verification policy from profile control defaults."""
    control_defaults = {} if profile is None else dict(profile.control_defaults)
    staged_enabled = (
        bool(control_defaults.get("staged_verification", False))
        if staged_verification is None
        else bool(staged_verification)
    )
    if not staged_enabled:
        return None
    default_policy = StagedVerificationPolicy()
    return StagedVerificationPolicy(
        verify_risk_levels=control_defaults.get(
            "stage_verify_risk_levels",
            default_policy.verify_risk_levels,
        ),
        verify_actions=control_defaults.get(
            "stage_verify_actions",
            default_policy.verify_actions,
        ),
        verify_claim_feature_flags=control_defaults.get(
            "stage_verify_claim_feature_flags",
            default_policy.verify_claim_feature_flags,
        ),
        verify_claim_metadata_keys=control_defaults.get(
            "stage_verify_claim_metadata_keys",
            default_policy.verify_claim_metadata_keys,
        ),
    )


def build_verifier(
    facts: dict[str, Any] | None,
    evidence: list[Any] | None,
    refutations: dict[str, Any] | None,
    *,
    enable_calculator: bool = False,
    route_name: str | None = None,
) -> Verifier:
    """Build a deterministic verifier from exact-match facts or grounded evidence."""
    if evidence is not None or refutations is not None:
        evidence_documents = () if evidence is None else tuple(evidence)
        base_verifier: Verifier = GroundednessVerifier(evidence=evidence_documents, refutations=refutations or {})
        return _with_optional_calculator(base_verifier, enabled=enable_calculator, route_name=route_name)
    if facts is None:
        facts = {
            "Paris is the capital of France": VerificationStatus.SUPPORTED.value,
            "The moon is made of cheese": VerificationStatus.REFUTED.value,
        }
    normalized = {
        normalize_claim_text(text): VerificationStatus(str(status))
        for text, status in facts.items()
    }
    return _with_optional_calculator(
        InMemoryVerifier(facts=normalized),
        enabled=enable_calculator,
        route_name=route_name,
    )


def _with_optional_calculator(verifier: Verifier, *, enabled: bool, route_name: str | None = None) -> Verifier:
    fallback_route_name = _normalized_route_name(route_name) or "fallback"
    if not enabled and route_name is None:
        return verifier
    if not enabled:
        return RoutedVerifier((VerifierRoute(fallback_route_name, verifier, fallback=True),))
    return RoutedVerifier((
        VerifierRoute(
            "calculator",
            CalculatorVerifier(),
            metadata_keys=("calculation", "expression"),
            context_keys=("calculation", "expression"),
            text_patterns=(ARITHMETIC_TEXT_PATTERN,),
        ),
        VerifierRoute(fallback_route_name, verifier, fallback=True),
    ))


def _normalized_route_name(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _promotion_contract_path(raw_path: str | None) -> Path | None:
    if raw_path is not None:
        return Path(raw_path)
    return default_promotion_contract_path()


def _promotion_contract_metadata(
    contract: ProductPromotionContract | None,
    *,
    source: str | None,
    budget_enabled: bool,
) -> dict[str, Any]:
    if contract is None:
        return {
            "promotion_contract_source": None,
            "promotion_contract_budget_enabled": False,
        }
    return {
        "promotion_contract_source": source,
        "promotion_contract_budget_enabled": budget_enabled,
        "promotion_contract_model_id": contract.model_id,
        "promotion_contract_source_workflow": contract.source_workflow,
        "promotion_contract_source_status": contract.source_status,
        "promotion_contract_runtime": dict(contract.runtime),
        "promotion_contract_verifier_route": dict(contract.verifier_route),
        "promotion_contract_metadata": dict(contract.metadata),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run the calibrated-control demo and return the JSON-ready trace."""
    artifact = load_artifact(args.artifact)
    runtime_profile = get_runtime_profile(args.runtime_profile)
    explicit_promotion_contract = getattr(args, "promotion_contract", None) is not None
    promotion_contract_path = _promotion_contract_path(getattr(args, "promotion_contract", None))
    promotion_contract = (
        None
        if promotion_contract_path is None
        else ProductPromotionContract.from_json(promotion_contract_path)
    )
    setattr(args, "_promotion_contract", promotion_contract)
    stage_policy = stage_policy_from_runtime_profile(
        runtime_profile,
        staged_verification=args.staged_verification,
    )
    diagnostics = (
        default_diagnostics_for_artifact(artifact)
        if args.diagnostics is None
        else parse_json_mapping(args.diagnostics, name="--diagnostics")
    )
    facts = None if args.facts is None else parse_json_mapping(args.facts, name="--facts")
    evidence = None if args.evidence is None else parse_json_sequence(args.evidence, name="--evidence")
    refutations = None if args.refutations is None else parse_json_mapping(args.refutations, name="--refutations")
    calculator_context = (
        {}
        if args.calculator_context is None
        else parse_json_mapping(args.calculator_context, name="--calculator-context")
    )
    retrieval_evidence = (
        None
        if args.retrieval_evidence is None
        else parse_json_sequence(args.retrieval_evidence, name="--retrieval-evidence")
    )

    claims = extract_claims(args.text)
    verifier_route_name = (
        None
        if promotion_contract is None
        else _normalized_route_name(promotion_contract.verifier_route.get("route"))
    )
    verifier = build_verifier(
        facts,
        evidence,
        refutations,
        enable_calculator=args.enable_calculator,
        route_name=verifier_route_name,
    )
    verifier_cache = None
    if getattr(args, "cache_verifier", False):
        verifier_cache = CachedVerifier(verifier)
        verifier = verifier_cache
    controller = RiskController(artifact)
    executor_registry = ActionExecutorRegistry()
    cache_metadata: dict[str, Any] = {}
    retriever_cache = None
    if retrieval_evidence is not None:
        retriever = InMemoryRetriever(retrieval_evidence)
        if getattr(args, "cache_retriever", False):
            retriever_cache = CachedRetriever(retriever)
            retriever = retriever_cache
        executor_registry.register(
            ControlAction.RETRIEVE,
            RetrievalActionExecutor(retriever),
        )

    loop_result = run_verification_loop(
        request_id=args.request_id,
        diagnostics={key: float(value) for key, value in diagnostics.items()},
        claims=claims,
        verifier=verifier,
        controller=controller,
        executor_registry=executor_registry,
        context=calculator_context,
        stage_policy=stage_policy,
        profile_runtime=getattr(args, "runtime_trace", True),
        metadata={
            "artifact_model_id": artifact.model_id,
            "artifact_source": artifact_source(args.artifact),
            "artifact_target_layer": artifact.target_layer,
            "artifact_scores": artifact.score_names(),
            "source": "examples/calibrated_control_demo.py",
            **_promotion_contract_metadata(
                promotion_contract,
                source=None if promotion_contract_path is None else str(promotion_contract_path),
                budget_enabled=explicit_promotion_contract,
            ),
            **runtime_profile_metadata(runtime_profile),
            "staged_verification_enabled": stage_policy is not None,
            "verifier_type": type(verifier).__name__,
            "calculator_enabled": args.enable_calculator,
            "action_executor_type": "ActionExecutorRegistry",
            "registered_actions": tuple(action.value for action in executor_registry.executors),
            "cache": cache_metadata,
        },
    )
    trace = loop_result.trace
    if verifier_cache is not None:
        cache_metadata["verifier"] = verifier_cache.stats.to_dict()
    if retriever_cache is not None:
        cache_metadata["retriever"] = retriever_cache.stats.to_dict()
    cache_summary = trace.cache_summary()
    route_cost_summary = trace.verification_route_cost_summary()
    payload = trace.to_dict()
    if cache_metadata:
        payload["metadata"]["cache_summary"] = cache_summary
    payload["metadata"]["route_cost_summary"] = route_cost_summary
    runtime_budget_policy = runtime_budget_policy_from_args(args)
    runtime_budget = (
        None
        if runtime_budget_policy is None
        else evaluate_product_runtime_budget(trace, runtime_budget_policy)
    )
    if runtime_budget is not None:
        payload["metadata"]["runtime_budget"] = runtime_budget
    output_path = Path(args.output) if args.output else None
    if args.registry and output_path is None:
        output_path = Path(args.registry).with_name(f"{args.request_id}_trace.json")
    if output_path is not None:
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.registry:
        ArtifactRegistry.load_json(args.registry).record_trace(
            name=args.request_id,
            path=str(output_path) if output_path is not None else "stdout",
            version="0.4",
            metadata={
                "source": "examples/calibrated_control_demo.py",
                "loop_version": "0.4",
                **_promotion_contract_metadata(
                    promotion_contract,
                    source=None if promotion_contract_path is None else str(promotion_contract_path),
                    budget_enabled=explicit_promotion_contract,
                ),
                **runtime_profile_metadata(runtime_profile),
                "staged_verification_enabled": stage_policy is not None,
                "action_execution_summary": trace.action_execution_summary(),
                "runtime_summary": trace.runtime_summary(),
                "cache_summary": cache_summary,
                "route_cost_summary": route_cost_summary,
                "runtime_budget": runtime_budget,
                "verifier_type": type(verifier).__name__,
            },
        ).save_json()
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="EigenTruth calibrated control-plane demo")
    parser.add_argument("--artifact", default=None, help="optional CalibrationArtifact JSON path")
    parser.add_argument("--diagnostics", default=None,
                        help="diagnostics JSON object; defaults to values that cross artifact thresholds")
    parser.add_argument("--text", default=DEFAULT_TEXT, help="draft text to extract and verify claims from")
    parser.add_argument("--facts", default=None, help="optional exact-match facts JSON object")
    parser.add_argument("--evidence", default=None, help="optional groundedness evidence JSON list")
    parser.add_argument("--refutations", default=None, help="optional groundedness refutations JSON object")
    parser.add_argument("--retrieval-evidence", default=None, help="optional retrieval documents JSON list")
    parser.add_argument("--enable-calculator", action="store_true",
                        help="run CalculatorVerifier before the selected lexical verifier")
    parser.add_argument("--calculator-context", default=None,
                        help="optional calculator context JSON object, e.g. {'calculation': {...}}")
    parser.add_argument("--runtime-profile", default=None, choices=RUNTIME_PROFILE_NAMES,
                        help="optional control-plane profile: latency, balanced, or audit")
    parser.add_argument("--staged-verification", dest="staged_verification", action="store_true",
                        default=None,
                        help="force staged verification even without a runtime profile")
    parser.add_argument("--no-staged-verification", dest="staged_verification", action="store_false",
                        help="force full initial verification even when a runtime profile enables staging")
    parser.add_argument("--no-runtime-trace", dest="runtime_trace", action="store_false",
                        default=True,
                        help="omit runtime phase timings from ProductTrace output")
    parser.add_argument("--promotion-contract", default=None,
                        help="optional ProductPromotionContract or release-candidate report JSON path")
    parser.add_argument("--cache-verifier", action="store_true",
                        help="wrap the selected verifier in request-local CachedVerifier and report cache stats")
    parser.add_argument("--cache-retriever", action="store_true",
                        help="wrap the in-memory retriever in request-local CachedRetriever and report cache stats")
    parser.add_argument("--max-runtime-total-seconds", type=float, default=None,
                        help="optional ProductTrace runtime budget for total request seconds")
    parser.add_argument("--max-runtime-phase-seconds", default=None,
                        help="optional JSON object mapping runtime phase names to max seconds")
    parser.add_argument("--max-runtime-phase-p95-seconds", default=None,
                        help="optional JSON object mapping runtime phase names to max p95 seconds")
    parser.add_argument("--max-runtime-phase-p99-seconds", default=None,
                        help="optional JSON object mapping runtime phase names to max p99 seconds")
    parser.add_argument("--max-mean-route-duration-seconds", type=float, default=None,
                        help="optional ProductTrace route-cost budget for mean route seconds")
    parser.add_argument("--max-p95-route-duration-seconds", type=float, default=None,
                        help="optional ProductTrace route-cost budget for p95 route seconds")
    parser.add_argument("--max-p99-route-duration-seconds", type=float, default=None,
                        help="optional ProductTrace route-cost budget for p99 route seconds")
    parser.add_argument("--max-route-duration-seconds", type=float, default=None,
                        help="optional ProductTrace route-cost budget for max route seconds")
    parser.add_argument("--max-mean-attempted-route-count", type=float, default=None,
                        help="optional ProductTrace route-cost budget for mean attempted routes per claim")
    parser.add_argument("--max-retrieval-use-rate", type=float, default=None,
                        help="optional ProductTrace route-cost budget for retrieval use rate")
    parser.add_argument("--max-retrieval-hit-count", type=float, default=None,
                        help="optional ProductTrace route-cost budget for total retrieval hits")
    parser.add_argument("--min-cache-hit-rate", type=float, default=None,
                        help="optional ProductTrace cache budget for aggregate cache hit rate")
    parser.add_argument("--min-named-cache-hit-rate", default=None,
                        help="optional JSON object mapping cache names to minimum hit rates")
    parser.add_argument("--registry", default=None, help="optional local ArtifactRegistry JSON path")
    parser.add_argument("--request-id", default="demo-request", help="request id stored in the ProductTrace")
    parser.add_argument("--output", default=None, help="optional path to write the trace JSON")
    payload = run(parser.parse_args())
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
