"""Tests for the frontier-toolkit MVP modules."""

import pytest
import torch

from eigentruth.adapters import InMemoryWorldModelAdapter
from eigentruth.calibration import CalibrationArtifact, CalibrationScore
from eigentruth.control import ControlAction, RiskController, RiskLevel
from eigentruth.core import TruthSubspace
from eigentruth.verify import InMemoryVerifier, VerificationStatus, extract_claims, normalize_claim_text


def test_truth_subspace_residual_distance_separates_off_plane_state():
    states = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],
    ])
    subspace = TruthSubspace.fit(states, rank=1)

    on_plane = torch.tensor([[1.5, 0.0, 0.0]])
    off_plane = torch.tensor([[1.5, 2.0, 0.0]])

    assert subspace.is_ready()
    assert subspace.residual_distance(on_plane).item() == pytest.approx(0.0, abs=1e-6)
    assert subspace.residual_distance(off_plane).item() > 1.9


def test_truth_subspace_contrastive_projection():
    true_states = torch.tensor([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    false_states = torch.tensor([[-1.0, 0.0], [-2.0, 0.0]])
    subspace = TruthSubspace.fit_contrastive(true_states, false_states, rank=1)

    true_projection = subspace.truth_projection(torch.tensor([[2.0, 0.0]])).item()
    false_projection = subspace.truth_projection(torch.tensor([[-2.0, 0.0]])).item()

    assert true_projection > false_projection


def test_risk_controller_accepts_and_routes_threshold_exceedance():
    artifact = CalibrationArtifact(
        model_id="tiny",
        target_layer=-1,
        scores=(
            CalibrationScore("maha", threshold=3.0),
            CalibrationScore("support", threshold=0.4, direction="lower"),
        ),
        eigentruth_version="0.1.0",
    )
    controller = RiskController(artifact)

    low = controller.decide({"maha": 2.0, "support": 0.8})
    medium = controller.decide({"maha": 4.0, "support": 0.8})
    high = controller.decide({"maha": 4.0, "support": 0.1})

    assert low.action is ControlAction.ACCEPT
    assert low.risk_level is RiskLevel.LOW
    assert medium.action is ControlAction.RETRIEVE
    assert medium.risk_level is RiskLevel.MEDIUM
    assert high.action is ControlAction.ABSTAIN
    assert high.risk_level is RiskLevel.HIGH
    assert high.diagnostics["triggered_scores"] == ("maha", "support")


def test_claim_extraction_and_in_memory_verifier():
    text = "Paris is the capital of France. The moon is made of cheese!"
    claims = extract_claims(text)
    verifier = InMemoryVerifier(
        facts={
            normalize_claim_text("Paris is the capital of France"): VerificationStatus.SUPPORTED,
            normalize_claim_text("The moon is made of cheese"): VerificationStatus.REFUTED,
        },
        evidence={normalize_claim_text("Paris is the capital of France"): ("atlas",)},
    )

    results = verifier.verify_many(claims)

    assert len(claims) == 2
    assert claims[0].span is not None
    assert results[0].status is VerificationStatus.SUPPORTED
    assert results[0].evidence == ("atlas",)
    assert results[1].status is VerificationStatus.REFUTED


def test_in_memory_world_model_adapter_verifies_and_predicts_state():
    verifier = InMemoryVerifier({normalize_claim_text("Inventory is 10"): VerificationStatus.SUPPORTED})
    adapter = InMemoryWorldModelAdapter(verifier=verifier)
    claim = extract_claims("Inventory is 10.")[0]

    result = adapter.verify(claim)
    prediction = adapter.predict({"inventory": 10}, {"set": {"inventory": 8}})

    assert result.status is VerificationStatus.SUPPORTED
    assert prediction.state["inventory"] == 8
    assert "Inventory" in adapter.explain(claim)
