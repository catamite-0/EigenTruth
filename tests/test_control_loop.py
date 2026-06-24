"""Tests for the 0.4 verification/action/reverification loop."""

import json

from eigentruth.adapters import InMemoryRetriever, RetrievalActionExecutor
from eigentruth.calibration import CalibrationArtifact, CalibrationScore
from eigentruth.control import (
    ActionExecutionStatus,
    ActionExecutorRegistry,
    ActionResult,
    ControlAction,
    EvidenceBundle,
    RiskController,
    RiskLevel,
    StagedVerificationPolicy,
    evidence_bundle_from_action_results,
    run_verification_loop,
)
from eigentruth.verify import Claim, GroundednessVerifier, VerificationResult, VerificationStatus, extract_claims


def _artifact() -> CalibrationArtifact:
    return CalibrationArtifact(
        model_id="tiny",
        target_layer=-1,
        scores=(CalibrationScore("maha_last", threshold=3.0),),
        eigentruth_version="0.1.0",
    )


def _registry_with_retrieval(documents, *, min_overlap: float = 0.2) -> ActionExecutorRegistry:
    return ActionExecutorRegistry().register(
        ControlAction.RETRIEVE,
        RetrievalActionExecutor(InMemoryRetriever(documents, min_overlap=min_overlap)),
    )


def test_verification_loop_uses_retrieval_hits_for_final_accept():
    claims = extract_claims("Paris is the capital of France.")
    verifier = GroundednessVerifier(evidence=(), min_overlap=0.7)
    controller = RiskController(_artifact())
    registry = _registry_with_retrieval(("Paris is the capital of France.",))

    result = run_verification_loop(
        request_id="req-1",
        diagnostics={"maha_last": 1.0},
        claims=claims,
        verifier=verifier,
        controller=controller,
        executor_registry=registry,
    )

    assert result.initial_verification_results[0].status is VerificationStatus.INSUFFICIENT_EVIDENCE
    assert result.initial_decision.action is ControlAction.RETRIEVE
    assert result.action_results[0].status is ActionExecutionStatus.SUCCEEDED
    assert result.retrieval_evidence.has_evidence()
    assert result.final_verification_results[0].status is VerificationStatus.SUPPORTED
    assert result.final_decision.action is ControlAction.ACCEPT
    assert result.final_decision.risk_level is RiskLevel.LOW

    trace = result.trace.to_dict()
    assert trace["risk_decision"]["action"] == "accept"
    assert trace["verification_results"][0]["status"] == "supported"
    assert trace["events"][-1]["event_type"] == "final_risk_decision"
    runtime_trace = trace["runtime_trace"]
    assert runtime_trace is not None
    phase_names = {phase["name"] for phase in runtime_trace["phases"]}
    assert "initial_verification" in phase_names
    assert "action_execution" in phase_names
    assert "final_verification" in phase_names
    assert runtime_trace["summary"]["phase_counts"]["action_execution"] == 1
    assert runtime_trace["summary"]["total_seconds"] >= runtime_trace["summary"]["accounted_seconds"]
    json.dumps(result.to_dict())


def test_verification_loop_maps_retrieval_hits_to_fallback_claim_ids():
    claims = (Claim("Paris is the capital of France."),)
    verifier = GroundednessVerifier(evidence=(), min_overlap=0.7)
    registry = _registry_with_retrieval(("Paris is the capital of France.",))

    result = run_verification_loop(
        diagnostics={"maha_last": 1.0},
        claims=claims,
        verifier=verifier,
        controller=RiskController(_artifact()),
        executor_registry=registry,
    )

    assert result.action_requests[0].payload["retrieval_targets"][0]["claim_id"] == "c1"
    assert result.retrieval_evidence.to_dict()["claim_ids"] == ("c1",)
    assert result.final_verification_results[0].status is VerificationStatus.SUPPORTED


def test_verification_loop_keeps_retrieve_decision_when_no_hits_are_found():
    claims = extract_claims("Paris is the capital of France.")
    verifier = GroundednessVerifier(evidence=(), min_overlap=0.7)
    controller = RiskController(_artifact())
    registry = _registry_with_retrieval((), min_overlap=0.95)

    result = run_verification_loop(
        request_id="req-2",
        diagnostics={"maha_last": 1.0},
        claims=claims,
        verifier=verifier,
        controller=controller,
        executor_registry=registry,
    )

    assert result.action_results[0].status is ActionExecutionStatus.SUCCEEDED
    assert result.retrieval_evidence.has_evidence() is False
    assert result.final_verification_results[0].status is VerificationStatus.INSUFFICIENT_EVIDENCE
    assert result.final_decision.action is ControlAction.RETRIEVE
    assert result.final_decision.risk_level is RiskLevel.MEDIUM
    assert result.trace.to_dict()["risk_decision"]["action"] == "retrieve"


def test_verification_loop_can_disable_runtime_trace():
    claims = extract_claims("Paris is the capital of France.")
    verifier = GroundednessVerifier(evidence=("Paris is the capital of France.",), min_overlap=0.7)

    result = run_verification_loop(
        diagnostics={"maha_last": 1.0},
        claims=claims,
        verifier=verifier,
        controller=RiskController(_artifact()),
        profile_runtime=False,
    )

    assert result.trace.runtime_trace is None
    assert result.trace.runtime_summary()["measured_phases"] == 0
    assert result.trace.to_dict()["runtime_trace"] is None


def test_verification_loop_preserves_base_context_evidence_when_retrieval_is_claim_scoped():
    claims = extract_claims("Paris is the capital of France. Berlin is the capital of Germany.")
    verifier = GroundednessVerifier(evidence=(), min_overlap=0.7)
    registry = _registry_with_retrieval(("Berlin is the capital of Germany.",))

    result = run_verification_loop(
        diagnostics={"maha_last": 1.0},
        claims=claims,
        verifier=verifier,
        controller=RiskController(_artifact()),
        executor_registry=registry,
        context={"evidence": ({"text": "Paris is the capital of France.", "source": "atlas"},)},
    )

    assert [item.status for item in result.initial_verification_results] == [
        VerificationStatus.SUPPORTED,
        VerificationStatus.INSUFFICIENT_EVIDENCE,
    ]
    assert [item.status for item in result.final_verification_results] == [
        VerificationStatus.SUPPORTED,
        VerificationStatus.SUPPORTED,
    ]
    assert result.final_decision.action is ControlAction.ACCEPT
    assert result.final_decision.risk_level is RiskLevel.LOW


def test_verification_loop_does_not_override_refuted_claim_with_retrieval():
    claims = extract_claims("The moon is made of cheese.")
    verifier = GroundednessVerifier(
        evidence=(),
        refutations={"The moon is made of cheese": ("lunar samples are rock",)},
    )
    controller = RiskController(_artifact())
    registry = _registry_with_retrieval(("The moon is made of cheese.",))

    result = run_verification_loop(
        request_id="req-3",
        diagnostics={"maha_last": 1.0},
        claims=claims,
        verifier=verifier,
        controller=controller,
        executor_registry=registry,
    )

    assert result.initial_verification_results[0].status is VerificationStatus.REFUTED
    assert result.action_requests[0].action is ControlAction.ABSTAIN
    assert result.action_results[0].status is ActionExecutionStatus.DRY_RUN
    assert result.retrieval_evidence.has_evidence() is False
    assert result.final_decision.action is ControlAction.ABSTAIN
    assert result.final_decision.risk_level is RiskLevel.HIGH


def test_staged_verification_loop_skips_expensive_verifier_for_low_risk_claim():
    claims = (Claim("Paris is a city.", claim_id="c1"),)
    verifier = _CountingVerifier()

    result = run_verification_loop(
        request_id="req-stage-low",
        diagnostics={"maha_last": 1.0},
        claims=claims,
        verifier=verifier,
        controller=RiskController(_artifact()),
        stage_policy=StagedVerificationPolicy(),
    )

    assert verifier.verify_many_calls == 0
    assert verifier.verify_calls == 0
    assert result.initial_verification_results == ()
    assert result.final_verification_results == ()
    assert result.initial_decision.action is ControlAction.ACCEPT
    assert result.final_decision.action is ControlAction.ACCEPT
    assert result.verification_stage_decision is not None
    assert result.verification_stage_decision.run_verifier is False
    trace = result.trace.to_dict()
    assert trace["metadata"]["staged_verification"]["verify_risk_levels"] == (
        "medium",
        "high",
        "unknown",
    )
    assert trace["events"][1]["event_type"] == "verification_stage_decision"
    assert trace["events"][1]["payload"]["run_verifier"] is False
    assert trace["events"][2]["event_type"] == "initial_verification_skipped"
    assert trace["events"][3]["payload"]["skipped"] is True
    runtime_trace = trace["runtime_trace"]
    assert runtime_trace is not None
    phase_names = {phase["name"] for phase in runtime_trace["phases"]}
    assert "verification_stage_decision" in phase_names
    assert "initial_verification" not in phase_names
    json.dumps(result.to_dict())


def test_staged_verification_loop_runs_verifier_for_sensitive_claim_metadata():
    claims = extract_claims("As of 2026, AlphaCorp has 10 offices.")
    verifier = _CountingVerifier(status=VerificationStatus.SUPPORTED)

    result = run_verification_loop(
        request_id="req-stage-sensitive",
        diagnostics={"maha_last": 1.0},
        claims=claims,
        verifier=verifier,
        controller=RiskController(_artifact()),
        stage_policy=StagedVerificationPolicy(),
    )

    assert verifier.verify_many_calls == 1
    assert result.initial_verification_results[0].status is VerificationStatus.SUPPORTED
    assert result.final_decision.action is ControlAction.ACCEPT
    assert result.verification_stage_decision is not None
    assert result.verification_stage_decision.run_verifier is True
    assert result.verification_stage_decision.triggered_claim_ids == ("c1",)
    assert result.verification_stage_decision.triggered_features["c1"] == (
        "has_number",
        "is_time_sensitive",
    )


def test_staged_verification_policy_parses_string_feature_and_metadata_flags():
    controller = RiskController(_artifact())
    low_decision = controller.decide({"maha_last": 1.0})
    policy = StagedVerificationPolicy(
        verify_claim_feature_flags=("has_number",),
        verify_claim_metadata_keys=("requires_verification",),
    )

    string_true = policy.decide(
        low_decision,
        claims=(
            Claim(
                "Numeric claim.",
                claim_id="c1",
                metadata={"features": {"has_number": "true"}},
            ),
        ),
    )
    string_false = policy.decide(
        low_decision,
        claims=(
            Claim(
                "Explicitly low-risk claim.",
                claim_id="c2",
                metadata={
                    "features": {"has_number": "false"},
                    "requires_verification": "false",
                },
            ),
        ),
    )
    ambiguous = policy.decide(
        low_decision,
        claims=(
            Claim(
                "Ambiguous routing claim.",
                claim_id="c3",
                metadata={"requires_verification": "maybe"},
            ),
        ),
    )

    assert string_true.run_verifier is True
    assert string_true.triggered_features == {"c1": ("has_number",)}
    assert string_false.run_verifier is False
    assert ambiguous.run_verifier is True
    assert ambiguous.triggered_metadata == {"c3": ("requires_verification",)}


def test_staged_verification_loop_runs_verifier_for_diagnostic_risk():
    claims = (Claim("Paris is a city.", claim_id="c1"),)
    verifier = _CountingVerifier(status=VerificationStatus.SUPPORTED)

    result = run_verification_loop(
        request_id="req-stage-risk",
        diagnostics={"maha_last": 4.0},
        claims=claims,
        verifier=verifier,
        controller=RiskController(_artifact()),
        stage_policy=StagedVerificationPolicy(),
    )

    assert verifier.verify_many_calls == 1
    assert result.verification_stage_decision is not None
    assert result.verification_stage_decision.run_verifier is True
    assert result.verification_stage_decision.reason == "diagnostic risk level is medium"
    assert result.initial_verification_results[0].status is VerificationStatus.SUPPORTED


def test_evidence_bundle_from_action_results_preserves_claim_specific_context():
    action_result = ActionResult(
        action=ControlAction.RETRIEVE,
        status=ActionExecutionStatus.SUCCEEDED,
        output={
            "hits_by_query": (
                {
                    "query": {"query": "Paris capital", "claim_id": "c1"},
                    "hits": ({"text": "Paris is the capital of France.", "source": "atlas", "score": 0.9},),
                },
            ),
        },
    )

    bundle = evidence_bundle_from_action_results((action_result,))

    assert isinstance(bundle, EvidenceBundle)
    assert bundle.to_context("c1")["evidence"][0]["text"] == "Paris is the capital of France."
    assert bundle.to_context("c1")["evidence"][0]["metadata"]["claim_id"] == "c1"
    assert bundle.to_context("c2") == {"evidence": ()}
    assert bundle.to_dict()["claim_ids"] == ("c1",)


class _CountingVerifier:
    def __init__(self, status=VerificationStatus.INSUFFICIENT_EVIDENCE):
        self.status = status
        self.verify_calls = 0
        self.verify_many_calls = 0

    def verify(self, claim, context=None):
        del claim, context
        self.verify_calls += 1
        return VerificationResult(
            status=self.status,
            confidence=0.9,
            evidence=("counting verifier evidence",) if self.status is VerificationStatus.SUPPORTED else (),
            explanation="counting verifier",
        )

    def verify_many(self, claims, context=None):
        self.verify_many_calls += 1
        return tuple(self.verify(claim, context=context) for claim in claims)
