"""Calibrated control-plane demo.

This example does not load a language model. It shows the product-facing part of
EigenTruth's control workflow: load or create a calibration artifact, evaluate
diagnostics with `RiskController`, verify simple claims, execute actions through
`ActionExecutorRegistry`, and emit a JSON `ProductTrace`.

The output is a trace for routing and debugging. It is not proof that a response
is true, and the built-in thresholds are only toy values for demonstration.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from eigentruth.adapters import InMemoryRetriever, RetrievalActionExecutor
from eigentruth.calibration import CalibrationArtifact, CalibrationScore
from eigentruth.control import (
    ActionExecutorRegistry,
    ControlAction,
    RiskController,
    run_verification_loop,
)
from eigentruth.registry import ArtifactRegistry
from eigentruth.verify import (
    GroundednessVerifier,
    InMemoryVerifier,
    VerificationStatus,
    Verifier,
    extract_claims,
    normalize_claim_text,
)

DEFAULT_TEXT = "Paris is the capital of France. The moon is made of cheese."
DEFAULT_DIAGNOSTICS = {"maha_last": 4.2, "subspace_resid": 0.4}


def default_artifact() -> CalibrationArtifact:
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


def load_artifact(path: str | None) -> CalibrationArtifact:
    """Load a calibration artifact or return the built-in demo artifact."""
    if path is None:
        return default_artifact()
    return CalibrationArtifact.load_json(path)


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


def build_verifier(
    facts: dict[str, Any] | None,
    evidence: list[Any] | None,
    refutations: dict[str, Any] | None,
) -> Verifier:
    """Build a deterministic verifier from exact-match facts or grounded evidence."""
    if evidence is not None or refutations is not None:
        evidence_documents = () if evidence is None else tuple(evidence)
        return GroundednessVerifier(evidence=evidence_documents, refutations=refutations or {})
    if facts is None:
        facts = {
            "Paris is the capital of France": VerificationStatus.SUPPORTED.value,
            "The moon is made of cheese": VerificationStatus.REFUTED.value,
        }
    normalized = {
        normalize_claim_text(text): VerificationStatus(str(status))
        for text, status in facts.items()
    }
    return InMemoryVerifier(facts=normalized)


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run the calibrated-control demo and return the JSON-ready trace."""
    artifact = load_artifact(args.artifact)
    diagnostics = parse_json_mapping(args.diagnostics, name="--diagnostics")
    facts = None if args.facts is None else parse_json_mapping(args.facts, name="--facts")
    evidence = None if args.evidence is None else parse_json_sequence(args.evidence, name="--evidence")
    refutations = None if args.refutations is None else parse_json_mapping(args.refutations, name="--refutations")
    retrieval_evidence = (
        None
        if args.retrieval_evidence is None
        else parse_json_sequence(args.retrieval_evidence, name="--retrieval-evidence")
    )

    claims = extract_claims(args.text)
    verifier = build_verifier(facts, evidence, refutations)
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
        metadata={
            "artifact_model_id": artifact.model_id,
            "artifact_target_layer": artifact.target_layer,
            "artifact_scores": artifact.score_names(),
            "source": "examples/calibrated_control_demo.py",
            "verifier_type": type(verifier).__name__,
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
                "action_execution_summary": trace.action_execution_summary(),
                "verifier_type": type(verifier).__name__,
            },
        ).save_json()
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="EigenTruth calibrated control-plane demo")
    parser.add_argument("--artifact", default=None, help="optional CalibrationArtifact JSON path")
    parser.add_argument("--diagnostics", default=json.dumps(DEFAULT_DIAGNOSTICS), help="diagnostics JSON object")
    parser.add_argument("--text", default=DEFAULT_TEXT, help="draft text to extract and verify claims from")
    parser.add_argument("--facts", default=None, help="optional exact-match facts JSON object")
    parser.add_argument("--evidence", default=None, help="optional groundedness evidence JSON list")
    parser.add_argument("--refutations", default=None, help="optional groundedness refutations JSON object")
    parser.add_argument("--retrieval-evidence", default=None, help="optional retrieval documents JSON list")
    parser.add_argument("--registry", default=None, help="optional local ArtifactRegistry JSON path")
    parser.add_argument("--request-id", default="demo-request", help="request id stored in the ProductTrace")
    parser.add_argument("--output", default=None, help="optional path to write the trace JSON")
    payload = run(parser.parse_args())
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
