"""Tests for the frontier-toolkit MVP modules."""

import pytest
import torch

from eigentruth.adapters import InMemoryWorldModelAdapter
from eigentruth.calibration import CalibrationArtifact, CalibrationScore
from eigentruth.control import ControlAction, RiskController, RiskLevel
from eigentruth.core import TruthSubspace
from eigentruth.verify import (
    EvidenceDocument,
    GroundednessVerifier,
    InMemoryVerifier,
    VerificationResult,
    VerificationStatus,
    extract_claims,
    normalize_claim_text,
)


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


def test_risk_controller_combines_diagnostics_and_verification_results():
    artifact = CalibrationArtifact(
        model_id="tiny",
        target_layer=-1,
        scores=(CalibrationScore("maha", threshold=3.0),),
        eigentruth_version="0.1.0",
    )
    controller = RiskController(artifact)

    supported = (VerificationResult(VerificationStatus.SUPPORTED, confidence=0.9),)
    unsupported = (VerificationResult(VerificationStatus.INSUFFICIENT_EVIDENCE, confidence=0.2),)
    refuted = (VerificationResult(VerificationStatus.REFUTED, confidence=0.92),)
    errored = ({"status": "unexpected_status", "confidence": "0.4"},)

    low_supported = controller.decide({"maha": 1.0}, verification_results=supported)
    low_unsupported = controller.decide({"maha": 1.0}, verification_results=unsupported)
    compound = controller.decide({"maha": 4.0}, verification_results=unsupported)
    high_refuted = controller.decide({"maha": 1.0}, verification_results=refuted)
    unknown_error = controller.decide({"maha": 1.0}, verification_results=errored)
    compound_error = controller.decide({"maha": 4.0}, verification_results=errored)

    assert low_supported.action is ControlAction.ACCEPT
    assert low_supported.diagnostics["verification"]["counts"]["supported"] == 1
    assert low_unsupported.action is ControlAction.RETRIEVE
    assert low_unsupported.risk_level is RiskLevel.MEDIUM
    assert compound.action is ControlAction.ABSTAIN
    assert compound.risk_level is RiskLevel.HIGH
    assert high_refuted.action is ControlAction.ABSTAIN
    assert high_refuted.risk_level is RiskLevel.HIGH
    assert high_refuted.confidence == pytest.approx(0.92)
    assert high_refuted.diagnostics["verification"]["triggered_statuses"] == ("refuted",)
    assert unknown_error.action is ControlAction.CLARIFY
    assert unknown_error.risk_level is RiskLevel.UNKNOWN
    assert compound_error.action is ControlAction.ABSTAIN
    assert compound_error.risk_level is RiskLevel.HIGH


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


def test_groundedness_verifier_supports_refutes_and_reports_evidence():
    verifier = GroundednessVerifier(
        evidence=(
            EvidenceDocument("Paris is the capital of France and appears in the reference atlas.", source="atlas"),
            EvidenceDocument("The moon is not made of cheese; lunar samples are rock.", source="nasa"),
        ),
        refutations={"Mars is the capital of France": ("atlas: Paris is the capital of France",)},
        min_overlap=0.55,
    )
    claims = extract_claims(
        "Paris is the capital of France. The moon is made of cheese. Mars is the capital of France."
    )

    results = verifier.verify_many(claims)

    assert results[0].status is VerificationStatus.SUPPORTED
    assert results[0].metadata["best_source"] == "atlas"
    assert results[1].status is VerificationStatus.REFUTED
    assert results[1].metadata["decision_rule"] == "negation_mismatch"
    assert results[2].status is VerificationStatus.REFUTED
    assert results[2].metadata["decision_rule"] == "configured_refutation"


def test_groundedness_verifier_returns_insufficient_evidence_for_low_overlap():
    verifier = GroundednessVerifier(
        evidence=({"text": "Paris is the capital of France.", "source": "atlas"},),
        min_overlap=0.8,
    )
    result = verifier.verify(extract_claims("Tokyo is the capital of Japan.")[0])

    assert result.status is VerificationStatus.INSUFFICIENT_EVIDENCE
    assert result.metadata["decision_rule"] == "low_overlap"
    assert result.metadata["best_overlap"] < 0.8


def test_in_memory_world_model_adapter_verifies_and_predicts_state():
    verifier = InMemoryVerifier({normalize_claim_text("Inventory is 10"): VerificationStatus.SUPPORTED})
    adapter = InMemoryWorldModelAdapter(verifier=verifier)
    claim = extract_claims("Inventory is 10.")[0]

    result = adapter.verify(claim)
    prediction = adapter.predict({"inventory": 10}, {"set": {"inventory": 8}})

    assert result.status is VerificationStatus.SUPPORTED
    assert prediction.state["inventory"] == 8
    assert "Inventory" in adapter.explain(claim)
