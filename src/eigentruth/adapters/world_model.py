"""Protocol for external world-model or domain-state correction adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, runtime_checkable

from eigentruth.verify import Claim, VerificationResult


@dataclass(frozen=True)
class WorldModelPrediction:
    """Predicted next state or constraint from a world/domain model."""

    state: Mapping[str, Any]
    confidence: float
    explanation: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("confidence must be in [0, 1].")


@runtime_checkable
class WorldModelAdapter(Protocol):
    """Adapter for stateful, causal, physical, or domain-specific verification."""

    def verify(self, claim: Claim, context: Mapping[str, Any] | None = None) -> VerificationResult:
        """Verify a claim against the world/domain model."""
        ...

    def predict(
        self,
        state: Mapping[str, Any],
        action: Mapping[str, Any],
    ) -> WorldModelPrediction:
        """Predict a state transition or domain consequence."""
        ...

    def explain(self, claim: Claim) -> str:
        """Explain the world-model basis for a claim-level judgment."""
        ...


@dataclass(frozen=True)
class InMemoryWorldModelAdapter:
    """Small world-model adapter for tests, demos, and deterministic rules.

    It verifies claims through an in-memory verifier and applies simple
    state updates from an action mapping. Concrete domain adapters can replace
    this with simulators, databases, or learned world models.
    """

    verifier: object

    def verify(self, claim: Claim, context: Mapping[str, Any] | None = None) -> VerificationResult:
        return self.verifier.verify(claim, context=context)

    def predict(
        self,
        state: Mapping[str, Any],
        action: Mapping[str, Any],
    ) -> WorldModelPrediction:
        next_state = dict(state)
        updates = action.get("set", action)
        if isinstance(updates, Mapping):
            next_state.update(updates)
        return WorldModelPrediction(
            state=next_state,
            confidence=1.0,
            explanation="in-memory state transition",
            metadata={"action": dict(action)},
        )

    def explain(self, claim: Claim) -> str:
        return f"in-memory world model checked claim: {claim.text}"
