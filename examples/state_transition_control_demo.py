"""State-transition world-model control demo.

This example does not load a language model. It demonstrates where a world
model fits in the product loop: internal diagnostics are low, but a deterministic
state-transition verifier predicts the consequence of an action and refutes a
claim about the resulting state, driving a dry-run ``abstain`` action.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from eigentruth.adapters import InMemoryWorldModelAdapter, StateTransitionVerifier, StructuredStateVerifier
from eigentruth.calibration import CalibrationArtifact, CalibrationScore
from eigentruth.control import ActionExecutorRegistry, RiskController, run_verification_loop
from eigentruth.verify import Claim


def demo_artifact() -> CalibrationArtifact:
    """Return a toy artifact whose diagnostic is below threshold by default."""
    return CalibrationArtifact(
        model_id="state-transition-demo",
        target_layer=-1,
        scores=(CalibrationScore("truth_proj", threshold=1.0, direction="higher", conformal_alpha=0.1),),
        eigentruth_version="0.1.0",
        calibration_dataset_metadata={
            "source": "examples/state_transition_control_demo.py",
            "note": "toy threshold for product-control demonstration only",
        },
    )


def default_state() -> dict[str, Any]:
    """Return deterministic order state for the demo."""
    return {
        "inventory": {"sku_123": {"available": 10}},
        "orders": {"ord_1": {"status": "pending", "quantity": 3}},
    }


def default_diagnostics() -> dict[str, float]:
    """Return low-risk diagnostics so verifier output drives the decision."""
    return {"truth_proj": 0.0}


def demo_claims() -> tuple[Claim, ...]:
    """Return action-conditioned claims with structured postconditions."""
    action = {
        "decrement": {"inventory.sku_123.available": 3},
        "set": {"orders.ord_1.status": "reserved"},
    }
    return (
        Claim(
            "After reserving order ord_1, SKU 123 has 7 units available.",
            claim_id="transition-supported",
            metadata={
                "state_transition": {
                    "action": action,
                    "postcondition": {
                        "path": "inventory.sku_123.available",
                        "operator": "eq",
                        "value": 7,
                        "source": "in_memory_order_world",
                    },
                }
            },
        ),
        Claim(
            "After reserving order ord_1, SKU 123 still has 10 units available.",
            claim_id="transition-refuted",
            metadata={
                "state_transition": {
                    "action": action,
                    "postcondition": {
                        "path": "inventory.sku_123.available",
                        "operator": "eq",
                        "value": 10,
                        "source": "in_memory_order_world",
                    },
                }
            },
        ),
    )


def parse_json_object(value: str | None, *, default: dict[str, Any], name: str) -> dict[str, Any]:
    """Parse a JSON object argument or return the provided default."""
    if value is None:
        return dict(default)
    payload = json.loads(value)
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must be a JSON object.")
    return dict(payload)


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run the state-transition control demo and return a JSON-ready trace."""
    artifact = demo_artifact()
    diagnostics = {str(key): float(value) for key, value in parse_json_object(
        args.diagnostics,
        default=default_diagnostics(),
        name="--diagnostics",
    ).items()}
    state = parse_json_object(args.state, default=default_state(), name="--state")
    min_world_model_confidence = float(getattr(args, "min_world_model_confidence", 0.0))
    if not (0.0 <= min_world_model_confidence <= 1.0):
        raise ValueError("--min-world-model-confidence must be in [0, 1].")
    world_model = InMemoryWorldModelAdapter(verifier=StructuredStateVerifier(state={}))
    verifier = StateTransitionVerifier(
        world_model=world_model,
        state=state,
        min_prediction_confidence=min_world_model_confidence,
    )
    claims = demo_claims()
    loop_result = run_verification_loop(
        request_id=args.request_id,
        diagnostics=diagnostics,
        claims=claims,
        verifier=verifier,
        controller=RiskController(artifact),
        executor_registry=ActionExecutorRegistry(),
        metadata={
            "source": "examples/state_transition_control_demo.py",
            "artifact_model_id": artifact.model_id,
            "verifier_type": type(verifier).__name__,
            "world_model_type": type(world_model).__name__,
            "min_world_model_confidence": min_world_model_confidence,
            "business_domain": "order_fulfillment_transition",
        },
    )
    payload = loop_result.trace.to_dict()
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="EigenTruth state-transition control demo")
    parser.add_argument("--diagnostics", default=None,
                        help="diagnostics JSON object; defaults below the toy threshold")
    parser.add_argument("--state", default=None, help="base state JSON object; defaults to the order fixture")
    parser.add_argument("--min-world-model-confidence", type=float, default=0.0,
                        help="minimum prediction confidence required for world-model postcondition checks")
    parser.add_argument("--request-id", default="state-transition-demo", help="request id stored in ProductTrace")
    parser.add_argument("--output", default=None, help="optional path to write the trace JSON")
    payload = run(parser.parse_args())
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
