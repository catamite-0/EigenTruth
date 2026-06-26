"""Protocol and adapters for world-model or domain-state correction."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from eigentruth.adapters.state import StateCheck, StructuredStateVerifier
from eigentruth.verify import Claim, VerificationResult, VerificationStatus, stable_cache_key


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
class StateTransitionCheck:
    """One action-conditioned postcondition over predicted next state."""

    action: Mapping[str, Any]
    postcondition: StateCheck | Mapping[str, Any]
    source: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "StateTransitionCheck":
        """Build a transition check from a JSON-like mapping."""
        raw_action = data.get("action")
        if not isinstance(raw_action, Mapping):
            raise ValueError("state transition mapping must contain action object.")
        raw_postcondition = data.get("postcondition", data.get("state_check", data.get("check")))
        if not isinstance(raw_postcondition, (Mapping, StateCheck)):
            raise ValueError("state transition mapping must contain postcondition/state_check object.")
        return cls(
            action=dict(raw_action),
            postcondition=raw_postcondition,
            source=None if data.get("source") is None else str(data.get("source")),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(frozen=True)
class StateTransitionVerifier:
    """Verify claims about action consequences using a world-model adapter.

    The verifier reads `state_transition` metadata from the claim or context,
    predicts the next state with the configured world model, then evaluates a
    structured postcondition over that predicted state. Set
    `min_prediction_confidence` above zero to fail closed when the world-model
    prediction is too uncertain to support a postcondition judgment.
    """

    world_model: WorldModelAdapter
    state: Mapping[str, Any] = field(default_factory=dict)
    min_prediction_confidence: float = 0.0

    def __post_init__(self) -> None:
        if not (0.0 <= self.min_prediction_confidence <= 1.0):
            raise ValueError("min_prediction_confidence must be in [0, 1].")

    def verify(self, claim: Claim, context: Mapping[str, Any] | None = None) -> VerificationResult:
        """Verify one claim against a predicted state transition."""
        raw_check = _transition_check_source(claim, context)
        if raw_check is None:
            return VerificationResult(
                status=VerificationStatus.NOT_APPLICABLE,
                confidence=1.0,
                explanation="claim did not provide a state_transition",
                metadata={"verifier": "state_transition", "decision_rule": "no_state_transition"},
            )
        try:
            transition = (
                raw_check
                if isinstance(raw_check, StateTransitionCheck)
                else StateTransitionCheck.from_mapping(raw_check)
            )
            postcondition = (
                transition.postcondition
                if isinstance(transition.postcondition, StateCheck)
                else StateCheck.from_mapping(transition.postcondition)
            )
        except (TypeError, ValueError) as exc:
            return VerificationResult(
                status=VerificationStatus.ERROR,
                confidence=0.3,
                explanation=str(exc),
                metadata={"verifier": "state_transition", "decision_rule": "invalid_state_transition"},
            )

        base_state = _merged_state(self.state, context)
        try:
            prediction = self.world_model.predict(base_state, transition.action)
        except Exception as exc:  # pragma: no cover - defensive adapter boundary
            return VerificationResult(
                status=VerificationStatus.ERROR,
                confidence=0.25,
                explanation=f"world model prediction failed: {exc}",
                metadata={"verifier": "state_transition", "decision_rule": "prediction_error"},
            )
        if prediction.metadata.get("below_min_agreement") is True:
            return VerificationResult(
                status=VerificationStatus.INSUFFICIENT_EVIDENCE,
                confidence=prediction.confidence,
                explanation="world model prediction agreement below threshold",
                metadata={
                    "verifier": "state_transition",
                    "decision_rule": "prediction_agreement_below_threshold",
                    "world_model": type(self.world_model).__name__,
                    "action": dict(transition.action),
                    "prediction_confidence": prediction.confidence,
                    "min_prediction_confidence": self.min_prediction_confidence,
                    "prediction_explanation": prediction.explanation,
                    "prediction_metadata": dict(prediction.metadata),
                    "source": transition.source,
                },
            )
        if prediction.confidence < self.min_prediction_confidence:
            return VerificationResult(
                status=VerificationStatus.INSUFFICIENT_EVIDENCE,
                confidence=prediction.confidence,
                explanation="world model prediction confidence below threshold",
                metadata={
                    "verifier": "state_transition",
                    "decision_rule": "prediction_confidence_below_threshold",
                    "world_model": type(self.world_model).__name__,
                    "action": dict(transition.action),
                    "prediction_confidence": prediction.confidence,
                    "min_prediction_confidence": self.min_prediction_confidence,
                    "prediction_explanation": prediction.explanation,
                    "prediction_metadata": dict(prediction.metadata),
                    "source": transition.source,
                },
            )

        postcondition_claim = Claim(
            text=claim.text,
            claim_id=claim.claim_id,
            span=claim.span,
            metadata={"state_check": postcondition},
        )
        state_result = StructuredStateVerifier(state=prediction.state).verify(postcondition_claim)
        confidence = min(state_result.confidence, prediction.confidence)
        metadata = {
            **dict(state_result.metadata),
            "verifier": "state_transition",
            "world_model": type(self.world_model).__name__,
            "action": dict(transition.action),
            "prediction_confidence": prediction.confidence,
            "prediction_explanation": prediction.explanation,
            "prediction_metadata": dict(prediction.metadata),
            "source": transition.source,
        }
        if state_result.status is VerificationStatus.SUPPORTED:
            decision_rule = "transition_postcondition_passed"
            explanation = "predicted state transition supports postcondition"
        elif state_result.status is VerificationStatus.REFUTED:
            decision_rule = "transition_postcondition_failed"
            explanation = "predicted state transition refutes postcondition"
        else:
            decision_rule = f"transition_{state_result.metadata.get('decision_rule', 'undecided')}"
            explanation = state_result.explanation
        return VerificationResult(
            status=state_result.status,
            confidence=confidence,
            evidence=tuple(state_result.evidence),
            explanation=explanation,
            metadata={**metadata, "decision_rule": decision_rule},
        )

    def verify_many(
        self,
        claims: Sequence[Claim],
        context: Mapping[str, Any] | None = None,
    ) -> tuple[VerificationResult, ...]:
        """Verify multiple claims."""
        return tuple(self.verify(claim, context=context) for claim in claims)


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
        next_state = _deep_copy_mapping(state)
        if "set" in action and isinstance(action["set"], Mapping):
            _apply_updates(next_state, action["set"])
        if "increment" in action and isinstance(action["increment"], Mapping):
            _apply_delta_updates(next_state, action["increment"], sign=1.0)
        if "decrement" in action and isinstance(action["decrement"], Mapping):
            _apply_delta_updates(next_state, action["decrement"], sign=-1.0)
        if not any(key in action for key in ("set", "increment", "decrement")):
            _apply_updates(next_state, action)
        return WorldModelPrediction(
            state=next_state,
            confidence=1.0,
            explanation="in-memory state transition",
            metadata={"action": dict(action)},
        )

    def explain(self, claim: Claim) -> str:
        return f"in-memory world model checked claim: {claim.text}"


@dataclass(frozen=True)
class EnsembleWorldModelAdapter:
    """Aggregate multiple world-model adapters with disagreement-aware confidence.

    The adapter selects the majority predicted state by stable JSON fingerprint,
    degrades confidence by agreement rate, and marks predictions as
    `below_min_agreement` when too few members agree. `StateTransitionVerifier`
    treats that marker as insufficient evidence.
    """

    world_models: Sequence[WorldModelAdapter]
    min_agreement: float = 1.0

    def __post_init__(self) -> None:
        world_models = tuple(self.world_models)
        if not world_models:
            raise ValueError("world_models must contain at least one adapter.")
        if not (0.0 < self.min_agreement <= 1.0):
            raise ValueError("min_agreement must be in (0, 1].")
        object.__setattr__(self, "world_models", world_models)

    def verify(self, claim: Claim, context: Mapping[str, Any] | None = None) -> VerificationResult:
        results: list[tuple[int, WorldModelAdapter, VerificationResult]] = []
        errors: list[dict[str, Any]] = []
        for idx, world_model in enumerate(self.world_models):
            try:
                results.append((idx, world_model, world_model.verify(claim, context=context)))
            except Exception as exc:  # pragma: no cover - defensive adapter boundary
                errors.append({
                    "index": idx,
                    "world_model": type(world_model).__name__,
                    "error": str(exc),
                })
        if not results:
            return VerificationResult(
                status=VerificationStatus.ERROR,
                confidence=0.0,
                explanation="all world-model ensemble verifiers failed",
                metadata={
                    "verifier": "world_model_ensemble",
                    "decision_rule": "all_members_failed",
                    "member_count": len(self.world_models),
                    "errors": errors,
                },
            )

        status_counts = Counter(result.status for _, _, result in results)
        max_count = max(status_counts.values())
        winners = sorted(
            (status for status, count in status_counts.items() if count == max_count),
            key=lambda status: status.value,
        )
        member_count = len(self.world_models)
        agreement_rate = max_count / member_count
        metadata = {
            "verifier": "world_model_ensemble",
            "member_count": member_count,
            "result_count": len(results),
            "error_count": len(errors),
            "agreement_count": max_count,
            "agreement_rate": agreement_rate,
            "min_agreement": self.min_agreement,
            "member_statuses": [result.status.value for _, _, result in results],
            "member_confidences": [result.confidence for _, _, result in results],
            "member_world_models": [type(world_model).__name__ for _, world_model, _ in results],
            "errors": errors,
        }
        if len(winners) > 1:
            return VerificationResult(
                status=VerificationStatus.INSUFFICIENT_EVIDENCE,
                confidence=0.0,
                explanation="world-model ensemble verifier status tie",
                metadata={
                    **metadata,
                    "decision_rule": "status_tie",
                    "below_min_agreement": True,
                },
            )

        status = winners[0]
        agreeing = [result for _, _, result in results if result.status is status]
        mean_confidence = sum(result.confidence for result in agreeing) / len(agreeing)
        below_min_agreement = agreement_rate < self.min_agreement
        confidence = 0.0 if below_min_agreement else mean_confidence * agreement_rate
        if below_min_agreement:
            return VerificationResult(
                status=VerificationStatus.INSUFFICIENT_EVIDENCE,
                confidence=confidence,
                explanation="world-model ensemble verifier agreement below threshold",
                metadata={
                    **metadata,
                    "decision_rule": "status_agreement_below_threshold",
                    "below_min_agreement": True,
                    "consensus_status": status.value,
                },
            )

        evidence = tuple(item for result in agreeing for item in result.evidence)
        return VerificationResult(
            status=status,
            confidence=confidence,
            evidence=evidence,
            explanation="world-model ensemble verifier consensus",
            metadata={
                **metadata,
                "decision_rule": "status_consensus",
                "below_min_agreement": False,
                "consensus_status": status.value,
            },
        )

    def predict(
        self,
        state: Mapping[str, Any],
        action: Mapping[str, Any],
    ) -> WorldModelPrediction:
        predictions: list[tuple[int, WorldModelAdapter, WorldModelPrediction]] = []
        errors: list[dict[str, Any]] = []
        for idx, world_model in enumerate(self.world_models):
            try:
                predictions.append((idx, world_model, world_model.predict(state, action)))
            except Exception as exc:  # pragma: no cover - defensive adapter boundary
                errors.append({
                    "index": idx,
                    "world_model": type(world_model).__name__,
                    "error": str(exc),
                })
        member_count = len(self.world_models)
        if not predictions:
            return WorldModelPrediction(
                state=_deep_copy_mapping(state),
                confidence=0.0,
                explanation="all world-model ensemble predictions failed",
                metadata={
                    "world_model": "EnsembleWorldModelAdapter",
                    "decision_rule": "all_members_failed",
                    "member_count": member_count,
                    "prediction_count": 0,
                    "error_count": len(errors),
                    "errors": errors,
                    "agreement_count": 0,
                    "agreement_rate": 0.0,
                    "min_agreement": self.min_agreement,
                    "below_min_agreement": True,
                },
            )

        groups: dict[str, list[tuple[int, WorldModelAdapter, WorldModelPrediction]]] = {}
        for item in predictions:
            _, _, prediction = item
            groups.setdefault(stable_cache_key(prediction.state), []).append(item)
        ranked_groups = sorted(
            groups.items(),
            key=lambda pair: (
                -len(pair[1]),
                -sum(prediction.confidence for _, _, prediction in pair[1]) / len(pair[1]),
                pair[0],
            ),
        )
        consensus_key, consensus_group = ranked_groups[0]
        agreement_count = len(consensus_group)
        agreement_rate = agreement_count / member_count
        mean_confidence = sum(prediction.confidence for _, _, prediction in consensus_group) / agreement_count
        below_min_agreement = agreement_rate < self.min_agreement
        confidence = 0.0 if below_min_agreement else mean_confidence * agreement_rate
        consensus_prediction = consensus_group[0][2]
        metadata = {
            "world_model": "EnsembleWorldModelAdapter",
            "decision_rule": (
                "prediction_agreement_below_threshold"
                if below_min_agreement
                else "prediction_consensus"
            ),
            "member_count": member_count,
            "prediction_count": len(predictions),
            "error_count": len(errors),
            "agreement_count": agreement_count,
            "agreement_rate": agreement_rate,
            "min_agreement": self.min_agreement,
            "below_min_agreement": below_min_agreement,
            "disagreement": agreement_count < member_count,
            "consensus_state_fingerprint": consensus_key,
            "member_confidences": [prediction.confidence for _, _, prediction in predictions],
            "member_world_models": [type(world_model).__name__ for _, world_model, _ in predictions],
            "errors": errors,
        }
        return WorldModelPrediction(
            state=_deep_copy_mapping(consensus_prediction.state),
            confidence=confidence,
            explanation=(
                "world-model ensemble agreement below threshold"
                if below_min_agreement
                else "world-model ensemble consensus"
            ),
            metadata=metadata,
        )

    def explain(self, claim: Claim) -> str:
        return f"world-model ensemble checked claim with {len(self.world_models)} members: {claim.text}"


def _transition_check_source(
    claim: Claim,
    context: Mapping[str, Any] | None,
) -> Mapping[str, Any] | StateTransitionCheck | None:
    if context is not None:
        by_claim = context.get("state_transitions")
        if isinstance(by_claim, Mapping) and claim.claim_id is not None and claim.claim_id in by_claim:
            candidate = by_claim[claim.claim_id]
            if isinstance(candidate, (Mapping, StateTransitionCheck)):
                return candidate
        direct = context.get("state_transition")
        if isinstance(direct, (Mapping, StateTransitionCheck)):
            return direct
    metadata = claim.metadata if isinstance(claim.metadata, Mapping) else {}
    direct = metadata.get("state_transition")
    if isinstance(direct, (Mapping, StateTransitionCheck)):
        return direct
    if "action" in metadata and any(key in metadata for key in ("postcondition", "state_check", "check")):
        return metadata
    return None


def _merged_state(base_state: Mapping[str, Any], context: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if context is None or not isinstance(context.get("state"), Mapping):
        return base_state
    merged = _deep_copy_mapping(base_state)
    _apply_updates(merged, context["state"])
    return merged


def _deep_copy_mapping(data: Mapping[str, Any]) -> dict[str, Any]:
    copied: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, Mapping):
            copied[key] = _deep_copy_mapping(value)
        elif isinstance(value, list):
            copied[key] = [_deep_copy_mapping(item) if isinstance(item, Mapping) else item for item in value]
        else:
            copied[key] = value
    return copied


def _apply_updates(state: dict[str, Any], updates: Mapping[str, Any]) -> None:
    for key, value in updates.items():
        if "." in str(key):
            _set_path(state, str(key), value)
        elif isinstance(value, Mapping) and isinstance(state.get(key), Mapping):
            target = _deep_copy_mapping(state[key])
            _apply_updates(target, value)
            state[str(key)] = target
        elif isinstance(value, Mapping):
            state[str(key)] = _deep_copy_mapping(value)
        else:
            state[str(key)] = value


def _set_path(state: dict[str, Any], path: str, value: Any) -> None:
    current = state
    parts = path.split(".")
    for part in parts[:-1]:
        existing = current.get(part)
        if existing is None:
            nested: dict[str, Any] = {}
            current[part] = nested
            current = nested
        elif isinstance(existing, Mapping):
            if not isinstance(existing, dict):
                existing = dict(existing)
                current[part] = existing
            current = existing
        else:
            raise ValueError(f"state path collision at {part!r} while setting {path!r}.")
    leaf = parts[-1]
    current[leaf] = _deep_copy_mapping(value) if isinstance(value, Mapping) else value


def _get_path(state: Mapping[str, Any], path: str) -> tuple[bool, Any]:
    current: Any = state
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return False, None
        current = current[part]
    return True, current


def _apply_delta_updates(state: dict[str, Any], updates: Mapping[str, Any], *, sign: float) -> None:
    for path, delta in updates.items():
        path_text = str(path)
        found, current = _get_path(state, path_text)
        if not found:
            raise ValueError(f"cannot apply delta to missing state path {path_text!r}.")
        try:
            next_value = float(current) + sign * float(delta)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"state delta for {path_text!r} requires numeric current and delta values.") from exc
        if isinstance(current, int) and isinstance(delta, int) and next_value.is_integer():
            _set_path(state, path_text, int(next_value))
        else:
            _set_path(state, path_text, next_value)
