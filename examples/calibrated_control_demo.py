"""Calibrated control-plane demo.

This example does not load a language model. It shows the product-facing part of
EigenTruth's control workflow: load or create a calibration artifact, evaluate
diagnostics with `RiskController`, verify simple claims, execute actions through
`ActionExecutorRegistry`, and emit a JSON `ProductTrace`.

The output is a trace for routing and debugging. It is not proof that a response
is true. When the repository's Qwen l80 calibration artifact is present, it is
used by default; otherwise the script falls back to toy thresholds.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from eigentruth.adapters import CalculatorVerifier, InMemoryRetriever, RetrievalActionExecutor
from eigentruth.calibration import CalibrationArtifact, CalibrationScore
from eigentruth.control import (
    RUNTIME_PROFILE_NAMES,
    ActionExecutorRegistry,
    ControlAction,
    RiskController,
    RuntimeProfile,
    StagedVerificationPolicy,
    get_runtime_profile,
    run_verification_loop,
)
from eigentruth.registry import ArtifactRegistry
from eigentruth.verify import (
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
DEFAULT_QWEN_ARTIFACT_PATH = (
    Path(__file__).resolve().parents[1] / "artifacts" / "qwen05_truthfulqa_l80_best_calibration.json"
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
    return DEFAULT_QWEN_ARTIFACT_PATH if DEFAULT_QWEN_ARTIFACT_PATH.exists() else None


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
) -> Verifier:
    """Build a deterministic verifier from exact-match facts or grounded evidence."""
    if evidence is not None or refutations is not None:
        evidence_documents = () if evidence is None else tuple(evidence)
        base_verifier: Verifier = GroundednessVerifier(evidence=evidence_documents, refutations=refutations or {})
        return _with_optional_calculator(base_verifier, enabled=enable_calculator)
    if facts is None:
        facts = {
            "Paris is the capital of France": VerificationStatus.SUPPORTED.value,
            "The moon is made of cheese": VerificationStatus.REFUTED.value,
        }
    normalized = {
        normalize_claim_text(text): VerificationStatus(str(status))
        for text, status in facts.items()
    }
    return _with_optional_calculator(InMemoryVerifier(facts=normalized), enabled=enable_calculator)


def _with_optional_calculator(verifier: Verifier, *, enabled: bool) -> Verifier:
    if not enabled:
        return verifier
    return RoutedVerifier((
        VerifierRoute(
            "calculator",
            CalculatorVerifier(),
            metadata_keys=("calculation", "expression"),
            context_keys=("calculation", "expression"),
            text_patterns=(ARITHMETIC_TEXT_PATTERN,),
        ),
        VerifierRoute("fallback", verifier, fallback=True),
    ))


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run the calibrated-control demo and return the JSON-ready trace."""
    artifact = load_artifact(args.artifact)
    runtime_profile = get_runtime_profile(args.runtime_profile)
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
    verifier = build_verifier(facts, evidence, refutations, enable_calculator=args.enable_calculator)
    controller = RiskController(artifact)
    executor_registry = ActionExecutorRegistry()
    if retrieval_evidence is not None:
        executor_registry.register(
            ControlAction.RETRIEVE,
            RetrievalActionExecutor(InMemoryRetriever(retrieval_evidence)),
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
        metadata={
            "artifact_model_id": artifact.model_id,
            "artifact_source": artifact_source(args.artifact),
            "artifact_target_layer": artifact.target_layer,
            "artifact_scores": artifact.score_names(),
            "source": "examples/calibrated_control_demo.py",
            **runtime_profile_metadata(runtime_profile),
            "staged_verification_enabled": stage_policy is not None,
            "verifier_type": type(verifier).__name__,
            "calculator_enabled": args.enable_calculator,
            "action_executor_type": "ActionExecutorRegistry",
            "registered_actions": tuple(action.value for action in executor_registry.executors),
        },
    )
    trace = loop_result.trace
    payload = trace.to_dict()
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
                **runtime_profile_metadata(runtime_profile),
                "staged_verification_enabled": stage_policy is not None,
                "action_execution_summary": trace.action_execution_summary(),
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
    parser.add_argument("--registry", default=None, help="optional local ArtifactRegistry JSON path")
    parser.add_argument("--request-id", default="demo-request", help="request id stored in the ProductTrace")
    parser.add_argument("--output", default=None, help="optional path to write the trace JSON")
    payload = run(parser.parse_args())
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
