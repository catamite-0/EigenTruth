"""Calibrated control-plane demo.

This example does not load a language model. It shows the product-facing part of
EigenTruth's control workflow: load or create a calibration artifact, evaluate
diagnostics with `RiskController`, verify simple claims, execute actions through
`ActionExecutorRegistry`, and emit a JSON `ProductTrace`.

The output is a trace for routing and debugging. It is not proof that a response
is true. When the repository's SmolLM2 l80 calibration artifact is present, it is
used by default; otherwise the script falls back to the Qwen l80 artifact and
then toy thresholds. When the SmolLM2 structured-retrieval-audit release
candidate is present, its product promotion contract supplies the default
verifier route, calibrated control defaults, adapter-family metadata, and
required-audit metadata. When that contract carries promoted release-efficiency
evidence, it also supplies the default runtime profile unless an explicit or
pre-generation profile is selected; pass it explicitly with
`--promotion-contract` to also enforce its runtime budget policy.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

from eigentruth.adapters import CachedRetriever, CalculatorVerifier, InMemoryRetriever, RetrievalActionExecutor
from eigentruth.calibration import CalibrationArtifact, CalibrationScore
from eigentruth.control import (
    RUNTIME_PROFILE_NAMES,
    ActionExecutorRegistry,
    ControlAction,
    ControlPolicyConfig,
    PreGenerationRiskAssessment,
    PreGenerationRiskPolicy,
    ProductPromotionContract,
    ProductRuntimeBudgetPolicy,
    RiskController,
    RuntimeProfile,
    RuntimeProfileSelection,
    RuntimeProfileSelectorPolicy,
    StagedVerificationPolicy,
    evaluate_product_runtime_budget,
    finalize_loop_answer,
    first_existing_product_promotion_contract_path,
    get_runtime_profile,
    load_product_runtime_evidence_bundle,
    product_promotion_contract_metadata,
    run_verification_loop,
    select_pre_generation_profile,
    select_runtime_profile,
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
DEFAULT_PROMOTION_CONTRACT_FILENAMES = (
    "smollm2_product_promotion_contract_v1_5/product-promotion-contract.json",
    (
        "smollm2_l20_inside_trigger_budget_derived_strict_structured_retrieval_audit_"
        "staged_release_candidate_v1_5_registry_workflow.json"
    ),
    (
        "smollm2_l20_inside_trigger_budget_derived_strict_structured_retrieval_audit_"
        "staged_release_candidate_v1_4_registry_workflow.json"
    ),
    (
        "smollm2_l20_inside_trigger_budget_derived_strict_structured_retrieval_audit_"
        "staged_release_candidate_registry_workflow.json"
    ),
    (
        "smollm2_l20_inside_trigger_budget_derived_structured_retrieval_audit_"
        "staged_release_candidate_registry_workflow.json"
    ),
    (
        "smollm2_l20_inside_trigger_budget_derived_retrieval_adapter_gated_"
        "staged_release_candidate_registry_workflow.json"
    ),
)
DEFAULT_PROMOTION_CONTRACT_PATHS = tuple(
    Path(__file__).resolve().parents[1]
    / "artifacts"
    / filename
    for filename in DEFAULT_PROMOTION_CONTRACT_FILENAMES
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
    return first_existing_product_promotion_contract_path(DEFAULT_PROMOTION_CONTRACT_PATHS)


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
    max_route_budget_exhaustion_rate = getattr(args, "max_route_budget_exhaustion_rate", None)
    max_retrieval_use_rate = getattr(args, "max_retrieval_use_rate", None)
    max_retrieval_hit_count = getattr(args, "max_retrieval_hit_count", None)
    min_cache_hit_rate = getattr(args, "min_cache_hit_rate", None)
    raw_named_cache_hit_rate = getattr(args, "min_named_cache_hit_rate", None)
    min_verification_skip_rate = getattr(args, "min_verification_skip_rate", None)
    min_selective_claim_skip_rate = getattr(args, "min_selective_claim_skip_rate", None)
    max_verified_claim_count = getattr(args, "max_verified_claim_count", None)
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
        and max_route_budget_exhaustion_rate is None
        and max_retrieval_use_rate is None
        and max_retrieval_hit_count is None
        and min_cache_hit_rate is None
        and raw_named_cache_hit_rate is None
        and min_verification_skip_rate is None
        and min_selective_claim_skip_rate is None
        and max_verified_claim_count is None
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
        max_route_budget_exhaustion_rate=(
            base_policy.max_route_budget_exhaustion_rate
            if max_route_budget_exhaustion_rate is None
            else max_route_budget_exhaustion_rate
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
        min_verification_skip_rate=(
            base_policy.min_verification_skip_rate
            if min_verification_skip_rate is None
            else min_verification_skip_rate
        ),
        min_selective_claim_skip_rate=(
            base_policy.min_selective_claim_skip_rate
            if min_selective_claim_skip_rate is None
            else min_selective_claim_skip_rate
        ),
        max_verified_claim_count=(
            base_policy.max_verified_claim_count
            if max_verified_claim_count is None
            else max_verified_claim_count
        ),
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


def runtime_profile_metadata(
    profile: RuntimeProfile | None,
    *,
    selection: RuntimeProfileSelection | None = None,
    selector_policy: RuntimeProfileSelectorPolicy | None = None,
    requested: str | None = None,
) -> dict[str, Any]:
    """Return trace metadata for the selected runtime profile."""
    if profile is None:
        return {
            "runtime_profile": None,
            "runtime_profile_control_defaults": None,
        }
    metadata = {
        "runtime_profile": profile.name,
        "runtime_profile_description": profile.description,
        "runtime_profile_control_defaults": dict(profile.control_defaults),
    }
    if selection is not None:
        metadata["runtime_profile_requested"] = requested
        metadata["runtime_profile_selection"] = selection.to_dict()
        metadata["runtime_profile_selector_policy"] = (
            None if selector_policy is None else selector_policy.to_dict()
        )
    return metadata


def stage_policy_from_runtime_profile(
    profile: RuntimeProfile | None,
    *,
    staged_verification: bool | None = None,
    control_defaults: Mapping[str, Any] | None = None,
) -> StagedVerificationPolicy | None:
    """Build a staged verification policy from profile control defaults."""
    control_defaults = (
        control_defaults_from_runtime_profile(profile)
        if control_defaults is None
        else dict(control_defaults)
    )
    staged_enabled = (
        _strict_bool(
            control_defaults.get("staged_verification", False),
            field_name="staged_verification",
        )
        if staged_verification is None
        else _strict_bool(staged_verification, field_name="staged_verification")
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
        verify_triggered_claims_only=control_defaults.get(
            "stage_verify_triggered_claims_only",
            default_policy.verify_triggered_claims_only,
        ),
        fail_closed_on_skip=control_defaults.get(
            "stage_fail_closed_on_skip",
            default_policy.fail_closed_on_skip,
        ),
    )


def _strict_bool(value: Any, *, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    raise ValueError(f"{field_name} must be a boolean")


def verifier_route_attempts_from_runtime_profile(
    profile: RuntimeProfile | None,
    *,
    max_verifier_route_attempts: int | None = None,
    control_defaults: Mapping[str, Any] | None = None,
) -> int | None:
    """Resolve the route-fanout cap from explicit args or profile defaults."""
    if max_verifier_route_attempts is not None:
        return _positive_int(max_verifier_route_attempts, name="max_verifier_route_attempts")
    control_defaults = (
        control_defaults_from_runtime_profile(profile)
        if control_defaults is None
        else dict(control_defaults)
    )
    value = control_defaults.get("max_verifier_route_attempts")
    return None if value is None else _positive_int(value, name="max_verifier_route_attempts")


def control_policy_config_from_promotion_contract(
    promotion_contract: ProductPromotionContract | None,
) -> ControlPolicyConfig | None:
    """Return the feedback-derived control policy from a promotion contract."""
    if promotion_contract is None or not promotion_contract.control_policy_config:
        return None
    return ControlPolicyConfig.from_dict(promotion_contract.control_policy_config)


def control_defaults_from_runtime_profile(
    profile: RuntimeProfile | None,
    *,
    promotion_contract: ProductPromotionContract | None = None,
) -> dict[str, Any]:
    """Return effective control defaults for the run.

    Runtime profiles provide local defaults. Promotion contracts can override
    those defaults when a release handoff artifact carries calibrated runtime
    recommendations. Explicit CLI/API parameters are applied by callers after
    this merge.
    """
    control_defaults = {} if profile is None else dict(profile.control_defaults)
    if promotion_contract is not None:
        control_defaults.update({
            str(key): value
            for key, value in promotion_contract.control_defaults.items()
            if value is not None
        })
    return control_defaults


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer.")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer.") from exc
    if parsed <= 0 or str(value).strip() not in {str(parsed), f"{parsed}.0"}:
        raise ValueError(f"{name} must be a positive integer.")
    return parsed


def resolve_runtime_profile(
    requested_profile: str | None,
    *,
    controller: RiskController,
    diagnostics: dict[str, float],
    claims: tuple[Any, ...],
    selector_policy: RuntimeProfileSelectorPolicy | None = None,
) -> tuple[RuntimeProfile | None, RuntimeProfileSelection | None]:
    """Resolve an explicit or automatic runtime profile request."""
    if requested_profile == "auto":
        selection = select_runtime_profile(
            controller.decide(diagnostics),
            claims=claims,
            selector_policy=selector_policy,
        )
        return get_runtime_profile(selection.selected_profile), selection
    return get_runtime_profile(requested_profile), None


def runtime_profile_from_release_efficiency(
    promotion_contract: ProductPromotionContract | None,
) -> str | None:
    """Return a promoted release-efficiency runtime profile recommendation."""
    if promotion_contract is None:
        return None
    release_efficiency = dict(promotion_contract.release_efficiency)
    status = release_efficiency.get("status")
    if status != "promote":
        return None
    raw_profile = release_efficiency.get("recommended_profile")
    if raw_profile is None:
        return None
    profile = str(raw_profile)
    allowed = {*RUNTIME_PROFILE_NAMES, "auto"}
    if profile not in allowed:
        allowed_values = ", ".join((*RUNTIME_PROFILE_NAMES, "auto"))
        raise ValueError(
            "promotion contract release_efficiency recommended_profile must be "
            f"one of: {allowed_values}."
        )
    return profile


def load_runtime_profile_selector_policy(path: str | None) -> RuntimeProfileSelectorPolicy | None:
    """Load an optional runtime-profile selector policy from JSON."""
    if path is None:
        return None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"runtime profile selector policy must be a JSON object: {path}")
    return RuntimeProfileSelectorPolicy.from_mapping(payload)


def load_pre_generation_risk_policy(path: str | None) -> PreGenerationRiskPolicy | None:
    """Load an optional pre-generation risk policy from JSON."""
    if path is None:
        return None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"pre-generation risk policy must be a JSON object: {path}")
    return PreGenerationRiskPolicy.from_mapping(payload)


def resolve_pre_generation_assessment(
    args: argparse.Namespace,
) -> tuple[PreGenerationRiskAssessment | None, PreGenerationRiskPolicy | None, dict[str, Any]]:
    """Return an optional pre-generation routing assessment from CLI-like args."""
    requested = getattr(args, "pre_generation_profile", "off")
    metadata = (
        {}
        if getattr(args, "pre_generation_metadata", None) is None
        else parse_json_mapping(args.pre_generation_metadata, name="--pre-generation-metadata")
    )
    policy = load_pre_generation_risk_policy(getattr(args, "pre_generation_risk_policy", None))
    if requested in (None, "off"):
        return None, policy, metadata
    if requested != "auto":
        raise ValueError("pre_generation_profile must be 'off' or 'auto'.")
    return (
        select_pre_generation_profile(
            getattr(args, "text", ""),
            metadata=metadata,
            risk_policy=policy,
        ),
        policy,
        metadata,
    )


def pre_generation_profile_metadata(
    assessment: PreGenerationRiskAssessment | None,
    *,
    policy: PreGenerationRiskPolicy | None,
    requested: str | None,
    metadata: dict[str, Any],
    runtime_profile_source: str | None,
) -> dict[str, Any]:
    """Return trace metadata for pre-generation runtime-profile routing."""
    return {
        "pre_generation_profile_requested": requested,
        "pre_generation_risk_assessment": None if assessment is None else assessment.to_dict(),
        "pre_generation_risk_policy": None if policy is None else policy.to_dict(),
        "pre_generation_metadata": dict(metadata),
        "runtime_profile_source": runtime_profile_source,
    }


def build_verifier(
    facts: dict[str, Any] | None,
    evidence: list[Any] | None,
    refutations: dict[str, Any] | None,
    *,
    enable_calculator: bool = False,
    route_name: str | None = None,
    max_attempted_routes: int | None = None,
) -> Verifier:
    """Build a deterministic verifier from exact-match facts or grounded evidence."""
    if evidence is not None or refutations is not None:
        evidence_documents = () if evidence is None else tuple(evidence)
        base_verifier: Verifier = GroundednessVerifier(evidence=evidence_documents, refutations=refutations or {})
        return _with_optional_calculator(
            base_verifier,
            enabled=enable_calculator,
            route_name=route_name,
            max_attempted_routes=max_attempted_routes,
        )
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
        max_attempted_routes=max_attempted_routes,
    )


def _with_optional_calculator(
    verifier: Verifier,
    *,
    enabled: bool,
    route_name: str | None = None,
    max_attempted_routes: int | None = None,
) -> Verifier:
    fallback_route_name = _normalized_route_name(route_name) or "fallback"
    if not enabled and route_name is None:
        return verifier
    if not enabled:
        return RoutedVerifier(
            (VerifierRoute(fallback_route_name, verifier, fallback=True),),
            max_attempted_routes=max_attempted_routes,
        )
    return RoutedVerifier(
        (
            VerifierRoute(
                "calculator",
                CalculatorVerifier(),
                metadata_keys=("calculation", "expression"),
                context_keys=("calculation", "expression"),
                text_patterns=(ARITHMETIC_TEXT_PATTERN,),
            ),
            VerifierRoute(fallback_route_name, verifier, fallback=True),
        ),
        max_attempted_routes=max_attempted_routes,
    )


def _normalized_route_name(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run the calibrated-control demo and return the JSON-ready trace."""
    artifact = load_artifact(args.artifact)
    explicit_promotion_contract = getattr(args, "promotion_contract", None) is not None
    runtime_evidence_bundle = load_product_runtime_evidence_bundle(
        getattr(args, "promotion_contract", None),
        default_contract_paths=DEFAULT_PROMOTION_CONTRACT_PATHS,
        manifest_path=getattr(args, "promotion_contract_manifest", None),
        registry_path=getattr(args, "promotion_contract_registry", None),
        registry_key=getattr(args, "promotion_contract_registry_key", None),
    )
    promotion_contract = None if runtime_evidence_bundle is None else runtime_evidence_bundle.contract
    promotion_contract_metadata = (
        product_promotion_contract_metadata(
            None,
            source=None,
            budget_enabled=False,
        )
        if runtime_evidence_bundle is None
        else runtime_evidence_bundle.runtime_metadata(
            budget_enabled=explicit_promotion_contract,
            verify_manifest=bool(getattr(args, "verify_promotion_contract_manifest", False)),
            verify_selfcheck_signal_fusion_manifest=bool(
                getattr(args, "verify_selfcheck_signal_fusion_manifest", False)
            ),
            include_selfcheck_signal_fusion_record=bool(
                getattr(args, "include_selfcheck_signal_fusion_record", False)
            ),
        )
    )
    setattr(args, "_promotion_contract", promotion_contract)
    diagnostics = (
        default_diagnostics_for_artifact(artifact)
        if args.diagnostics is None
        else parse_json_mapping(args.diagnostics, name="--diagnostics")
    )
    resolved_diagnostics = {key: float(value) for key, value in diagnostics.items()}
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
    control_policy_config = control_policy_config_from_promotion_contract(promotion_contract)
    control_policy_source = (
        None if control_policy_config is None else "promotion_contract_feedback_policy"
    )
    controller = RiskController(artifact, policy_config=control_policy_config)
    pre_generation_assessment, pre_generation_policy, pre_generation_metadata = (
        resolve_pre_generation_assessment(args)
    )
    selector_policy = load_runtime_profile_selector_policy(
        getattr(args, "runtime_profile_selector_policy", None)
    )
    requested_runtime_profile = getattr(args, "runtime_profile", None)
    effective_runtime_profile = requested_runtime_profile
    runtime_profile_source = None
    if effective_runtime_profile is None and pre_generation_assessment is not None:
        effective_runtime_profile = pre_generation_assessment.selected_profile
        runtime_profile_source = "pre_generation"
    elif effective_runtime_profile is None:
        effective_runtime_profile = runtime_profile_from_release_efficiency(
            promotion_contract
        )
        if effective_runtime_profile is not None:
            runtime_profile_source = "promotion_contract_release_efficiency"
    elif effective_runtime_profile == "auto":
        runtime_profile_source = "post_diagnostic_auto"
    elif effective_runtime_profile is not None:
        runtime_profile_source = "explicit"
    runtime_profile, runtime_profile_selection = resolve_runtime_profile(
        effective_runtime_profile,
        controller=controller,
        diagnostics=resolved_diagnostics,
        claims=claims,
        selector_policy=selector_policy,
    )
    runtime_control_defaults = control_defaults_from_runtime_profile(
        runtime_profile,
        promotion_contract=promotion_contract,
    )
    stage_policy = stage_policy_from_runtime_profile(
        runtime_profile,
        staged_verification=args.staged_verification,
        control_defaults=runtime_control_defaults,
    )
    max_verifier_route_attempts = verifier_route_attempts_from_runtime_profile(
        runtime_profile,
        max_verifier_route_attempts=getattr(args, "max_verifier_route_attempts", None),
        control_defaults=runtime_control_defaults,
    )
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
        max_attempted_routes=max_verifier_route_attempts,
    )
    verifier_cache = None
    if getattr(args, "cache_verifier", False):
        verifier_cache = CachedVerifier(verifier)
        verifier = verifier_cache
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
        diagnostics=resolved_diagnostics,
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
            **promotion_contract_metadata,
            **runtime_profile_metadata(
                runtime_profile,
                selection=runtime_profile_selection,
                selector_policy=selector_policy,
                requested=requested_runtime_profile,
            ),
            **pre_generation_profile_metadata(
                pre_generation_assessment,
                policy=pre_generation_policy,
                requested=getattr(args, "pre_generation_profile", "off"),
                metadata=pre_generation_metadata,
                runtime_profile_source=runtime_profile_source,
            ),
            "staged_verification_enabled": stage_policy is not None,
            "effective_control_policy_config": (
                None if control_policy_config is None else control_policy_config.to_dict()
            ),
            "control_policy_source": control_policy_source,
            "effective_control_defaults": runtime_control_defaults,
            "max_verifier_route_attempts": max_verifier_route_attempts,
            "verifier_type": type(verifier).__name__,
            "calculator_enabled": args.enable_calculator,
            "action_executor_type": "ActionExecutorRegistry",
            "registered_actions": tuple(action.value for action in executor_registry.executors),
            "cache": cache_metadata,
        },
    )
    final_answer = finalize_loop_answer(loop_result, draft_answer=args.text)
    trace = replace(loop_result.trace, final_answer=final_answer)
    if verifier_cache is not None:
        cache_metadata["verifier"] = verifier_cache.stats.to_dict()
    if retriever_cache is not None:
        cache_metadata["retriever"] = retriever_cache.stats.to_dict()
    cache_summary = trace.cache_summary()
    route_cost_summary = trace.verification_route_cost_summary()
    verification_stage_summary = trace.verification_stage_summary()
    final_answer_summary = trace.final_answer_summary()
    bounded_trace = bool(getattr(args, "bounded_trace", False))
    if bounded_trace:
        payload = trace.to_bounded_dict(
            max_claims=int(getattr(args, "bounded_trace_max_claims", 20)),
            max_verification_results=int(
                getattr(args, "bounded_trace_max_verification_results", 20)
            ),
            max_actions=int(getattr(args, "bounded_trace_max_actions", 20)),
            max_action_results=int(getattr(args, "bounded_trace_max_action_results", 20)),
            max_events=int(getattr(args, "bounded_trace_max_events", 20)),
            max_nested_items=int(getattr(args, "bounded_trace_max_nested_items", 16)),
            include_runtime_trace=bool(getattr(args, "bounded_trace_include_runtime_trace", False)),
        )
    else:
        payload = trace.to_dict()
    if cache_metadata:
        payload["metadata"]["cache_summary"] = (
            payload["summaries"]["cache"] if bounded_trace else cache_summary
        )
    payload["metadata"]["route_cost_summary"] = (
        payload["summaries"]["verification_route_cost"] if bounded_trace else route_cost_summary
    )
    payload["metadata"]["verification_stage_summary"] = (
        payload["summaries"]["verification_stage"] if bounded_trace else verification_stage_summary
    )
    payload["metadata"]["final_answer_summary"] = (
        payload["summaries"]["final_answer"] if bounded_trace else final_answer_summary
    )
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
        output_path.write_text(
            _json_text(payload, compact=bool(getattr(args, "compact_json", False))),
            encoding="utf-8",
        )
    if args.registry:
        ArtifactRegistry.load_json(args.registry).record_trace(
            name=args.request_id,
            path=str(output_path) if output_path is not None else "stdout",
            version="0.4",
            metadata={
                "source": "examples/calibrated_control_demo.py",
                "loop_version": "0.4",
                **promotion_contract_metadata,
                **runtime_profile_metadata(
                    runtime_profile,
                    selection=runtime_profile_selection,
                    selector_policy=selector_policy,
                    requested=requested_runtime_profile,
                ),
                **pre_generation_profile_metadata(
                    pre_generation_assessment,
                    policy=pre_generation_policy,
                    requested=getattr(args, "pre_generation_profile", "off"),
                    metadata=pre_generation_metadata,
                    runtime_profile_source=runtime_profile_source,
                ),
                "staged_verification_enabled": stage_policy is not None,
                "effective_control_policy_config": (
                    None if control_policy_config is None else control_policy_config.to_dict()
                ),
                "control_policy_source": control_policy_source,
                "effective_control_defaults": runtime_control_defaults,
                "max_verifier_route_attempts": max_verifier_route_attempts,
                "action_execution_summary": trace.action_execution_summary(),
                "runtime_summary": trace.runtime_summary(),
                "cache_summary": cache_summary,
                "route_cost_summary": route_cost_summary,
                "verification_stage_summary": verification_stage_summary,
                "final_answer_summary": final_answer_summary,
                "runtime_budget": runtime_budget,
                "verifier_type": type(verifier).__name__,
            },
        ).save_json()
    return payload


def _json_text(payload: Any, *, compact: bool) -> str:
    if compact:
        return json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


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
    parser.add_argument("--runtime-profile", default=None, choices=(*RUNTIME_PROFILE_NAMES, "auto"),
                        help="optional control-plane profile: latency, balanced, audit, or auto")
    parser.add_argument("--runtime-profile-selector-policy", default=None,
                        help="optional RuntimeProfileSelectorPolicy JSON path for --runtime-profile auto")
    parser.add_argument("--pre-generation-profile", default="off", choices=("off", "auto"),
                        help="optionally choose a runtime profile before generation from prompt metadata")
    parser.add_argument("--pre-generation-risk-policy", default=None,
                        help="optional PreGenerationRiskPolicy JSON path for --pre-generation-profile auto")
    parser.add_argument("--pre-generation-metadata", default=None,
                        help="optional JSON object of caller risk flags for pre-generation routing")
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
    parser.add_argument("--promotion-contract-manifest", default=None,
                        help="optional artifact manifest for the promotion contract")
    parser.add_argument("--promotion-contract-registry", default=None,
                        help="optional ArtifactRegistry JSON containing the promotion contract record")
    parser.add_argument("--promotion-contract-registry-key", default=None,
                        help="optional product_promotion_contract registry key")
    parser.add_argument("--verify-promotion-contract-manifest", action="store_true",
                        help="verify the promotion contract artifact manifest before recording metadata")
    parser.add_argument("--verify-selfcheck-signal-fusion-manifest", action="store_true",
                        help="verify the selfcheck-signal-fusion workflow manifest from the promotion contract")
    parser.add_argument("--include-selfcheck-signal-fusion-record", action="store_true",
                        help="attach the selfcheck-signal-fusion registry record referenced by the promotion contract")
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
    parser.add_argument("--max-route-budget-exhaustion-rate", type=float, default=None,
                        help="optional ProductTrace budget for capped verifier route exhaustion rate")
    parser.add_argument("--max-verifier-route-attempts", type=int, default=None,
                        help=("cap matching verifier routes attempted per claim; promotion contracts and "
                              "runtime profiles fill this when unset"))
    parser.add_argument("--max-retrieval-use-rate", type=float, default=None,
                        help="optional ProductTrace route-cost budget for retrieval use rate")
    parser.add_argument("--max-retrieval-hit-count", type=float, default=None,
                        help="optional ProductTrace route-cost budget for total retrieval hits")
    parser.add_argument("--min-cache-hit-rate", type=float, default=None,
                        help="optional ProductTrace cache budget for aggregate cache hit rate")
    parser.add_argument("--min-named-cache-hit-rate", default=None,
                        help="optional JSON object mapping cache names to minimum hit rates")
    parser.add_argument("--min-verification-skip-rate", type=float, default=None,
                        help="optional ProductTrace budget for staged-verification claim skip rate")
    parser.add_argument("--min-selective-claim-skip-rate", type=float, default=None,
                        help="optional ProductTrace budget for triggered-only staged-verification claim skip rate")
    parser.add_argument("--max-verified-claim-count", type=float, default=None,
                        help="optional ProductTrace budget for verifier-checked claim count")
    parser.add_argument("--registry", default=None, help="optional local ArtifactRegistry JSON path")
    parser.add_argument("--request-id", default="demo-request", help="request id stored in the ProductTrace")
    parser.add_argument("--output", default=None, help="optional path to write the trace JSON")
    parser.add_argument("--compact-json", action="store_true",
                        help="write minified ProductTrace JSON for automated artifact runs")
    parser.add_argument("--bounded-trace", action="store_true",
                        help="write a bounded ProductTrace payload with summaries and truncated lists")
    parser.add_argument("--bounded-trace-max-claims", type=int, default=20)
    parser.add_argument("--bounded-trace-max-verification-results", type=int, default=20)
    parser.add_argument("--bounded-trace-max-actions", type=int, default=20)
    parser.add_argument("--bounded-trace-max-action-results", type=int, default=20)
    parser.add_argument("--bounded-trace-max-events", type=int, default=20)
    parser.add_argument("--bounded-trace-max-nested-items", type=int, default=16)
    parser.add_argument("--bounded-trace-include-runtime-trace", action="store_true",
                        help="include full runtime_trace in bounded output instead of summaries only")
    parsed = parser.parse_args()
    payload = run(parsed)
    print(_json_text(payload, compact=bool(parsed.compact_json)), end="")


if __name__ == "__main__":
    main()
