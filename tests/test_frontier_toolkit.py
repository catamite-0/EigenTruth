"""Tests for the frontier-toolkit MVP modules."""

import math
import sqlite3
import time

import pytest
import torch

from eigentruth.adapters import (
    CachedRetriever,
    CachedStateSource,
    CalculatorVerifier,
    InMemoryRetriever,
    InMemoryWorldModelAdapter,
    QuestionAnswerFact,
    QuestionAnswerVerifier,
    RetrievalActionExecutor,
    RetrievalQuery,
    SQLiteFTSRetriever,
    SQLiteStateQuery,
    SQLiteStateSource,
    StateCheck,
    StateTransitionCheck,
    StateTransitionVerifier,
    StructuredStateVerifier,
    ToolOutputMapping,
    ToolOutputStateSource,
)
from eigentruth.calibration import CalibrationArtifact, CalibrationScore
from eigentruth.control import (
    RUNTIME_PROFILE_NAMES,
    RUNTIME_PROFILES,
    ActionExecutionPolicy,
    ActionExecutionStatus,
    ActionExecutorRegistry,
    ActionRequest,
    ActionResult,
    ControlAction,
    ControlPolicyConfig,
    DefaultCorrectionPolicy,
    DryRunActionExecutor,
    InMemoryActionExecutionLedger,
    PolicyGuardedActionExecutor,
    PreGenerationRiskAssessment,
    PreGenerationRiskPolicy,
    RiskController,
    RiskDecision,
    RiskLevel,
    RuntimeProfile,
    RuntimeProfileSelection,
    RuntimeProfileSelectorPolicy,
    TimeoutActionExecutor,
    get_runtime_profile,
    select_pre_generation_profile,
    select_runtime_profile,
)
from eigentruth.core import TruthSubspace
from eigentruth.verify import (
    CachedVerifier,
    Claim,
    ClaimDependency,
    ClaimVerificationPlan,
    CompositeVerifier,
    EvidenceDocument,
    EvidenceQualityPolicy,
    GroundednessVerifier,
    InMemoryVerifier,
    JsonTraceCache,
    RoutedVerifier,
    SelfConsistencyVerifier,
    VerificationResult,
    VerificationRouteHint,
    VerificationStatus,
    VerifierRoute,
    apply_claim_coherence,
    extract_calculation,
    extract_claims,
    infer_claim_dependencies,
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


def test_truth_subspace_rejects_single_factual_state():
    with pytest.raises(ValueError, match="at least two factual states"):
        TruthSubspace.fit(torch.tensor([[1.0, 2.0, 3.0]]), rank=1)


def test_truth_subspace_clamps_rank_to_centered_sample_rank():
    states = torch.tensor([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
    ])

    subspace = TruthSubspace.fit(states, rank=3)

    assert subspace.rank == 1
    assert subspace.basis.shape == (3, 1)


def test_truth_subspace_rejects_non_finite_states():
    states = torch.tensor([[1.0, 2.0], [float("nan"), 3.0]])

    with pytest.raises(ValueError, match="finite"):
        TruthSubspace.fit(states, rank=1)


def test_runtime_profiles_apply_only_missing_defaults():
    profile = get_runtime_profile("latency")

    assert isinstance(profile, RuntimeProfile)
    assert RUNTIME_PROFILE_NAMES == ("latency", "balanced", "audit")
    assert RUNTIME_PROFILES["balanced"].defaults["inside_trigger_budget_policy"] == "quality_balanced"
    assert RUNTIME_PROFILES["latency"].control_defaults["staged_verification"] is True
    assert RUNTIME_PROFILES["latency"].control_defaults["stage_verify_triggered_claims_only"] is True
    assert RUNTIME_PROFILES["latency"].control_defaults["max_verifier_route_attempts"] == 1
    assert RUNTIME_PROFILES["balanced"].control_defaults["max_verifier_route_attempts"] == 2
    assert RUNTIME_PROFILES["audit"].control_defaults["staged_verification"] is False

    merged, applied = profile.apply_defaults({
        "inside_trigger_budget_policy": None,
        "max_inside_sample_count_ratio": 0.5,
        "max_inside_generation_seconds_ratio": None,
        "max_mean_attempted_route_count": None,
        "max_retrieval_use_rate": None,
    })

    assert merged["inside_trigger_budget_policy"] == "cost_first"
    assert merged["max_inside_sample_count_ratio"] == pytest.approx(0.5)
    assert merged["max_inside_generation_seconds_ratio"] == pytest.approx(0.35)
    assert applied == {
        "inside_trigger_budget_policy": "cost_first",
        "max_inside_generation_seconds_ratio": 0.35,
        "max_mean_attempted_route_count": 1.1,
        "max_retrieval_use_rate": 0.0,
    }
    profile_payload = profile.to_dict()
    assert profile_payload["defaults"]["max_retrieval_use_rate"] == pytest.approx(0.0)
    assert profile_payload["control_defaults"]["stage_verify_risk_levels"] == ("high", "unknown")

    with pytest.raises(ValueError, match="runtime_profile"):
        get_runtime_profile("fast")


def test_select_runtime_profile_routes_by_risk_and_claim_metadata():
    low = RiskDecision(
        action=ControlAction.ACCEPT,
        risk_level=RiskLevel.LOW,
        confidence=1.0,
        reason="ok",
    )
    medium = RiskDecision(
        action=ControlAction.RETRIEVE,
        risk_level=RiskLevel.MEDIUM,
        confidence=0.7,
        reason="unsupported",
    )
    high = RiskDecision(
        action=ControlAction.ABSTAIN,
        risk_level=RiskLevel.HIGH,
        confidence=0.9,
        reason="risky",
    )

    low_selection = select_runtime_profile(
        low,
        claims=(Claim("Paris is the capital of France.", claim_id="c1", metadata={"features": {}}),),
    )
    sensitive_selection = select_runtime_profile(
        low,
        claims=(Claim("2 + 2 = 4.", claim_id="calc", metadata={"features": {"has_number": True}}),),
    )
    string_sensitive_selection = select_runtime_profile(
        low,
        claims=(
            Claim(
                "2 + 2 = 4.",
                claim_id="calc-string",
                metadata={"features": {"has_number": "true"}},
            ),
        ),
    )
    string_false_selection = select_runtime_profile(
        low,
        claims=(
            Claim(
                "Paris is the capital of France.",
                claim_id="false-string",
                metadata={"features": {"has_number": "false"}},
            ),
        ),
    )
    medium_selection = select_runtime_profile(medium, claims=())
    high_selection = select_runtime_profile(high, claims=())

    assert isinstance(low_selection, RuntimeProfileSelection)
    assert low_selection.selected_profile == "latency"
    assert low_selection.reason == "low diagnostic risk and no sensitive claim metadata"
    assert sensitive_selection.selected_profile == "audit"
    assert sensitive_selection.triggered_claim_ids == ("calc",)
    assert sensitive_selection.triggered_features == {"calc": ("has_number",)}
    assert string_sensitive_selection.selected_profile == "audit"
    assert string_sensitive_selection.triggered_features == {"calc-string": ("has_number",)}
    assert string_false_selection.selected_profile == "latency"
    assert medium_selection.selected_profile == "balanced"
    assert high_selection.selected_profile == "audit"


def test_runtime_profile_selector_policy_roundtrip_and_routes():
    low = RiskDecision(
        action=ControlAction.ACCEPT,
        risk_level=RiskLevel.LOW,
        confidence=1.0,
        reason="ok",
    )
    policy = RuntimeProfileSelectorPolicy.from_mapping({
        "sensitive_profile": "balanced",
        "sensitive_claim_feature_flags": ["has_citation"],
        "sensitive_claim_metadata_keys": ["requires_review"],
        "high_risk_actions": ["abstain"],
    })

    selection = select_runtime_profile(
        low,
        claims=(
            Claim(
                "2 + 2 = 4.",
                claim_id="calc",
                metadata={"features": {"has_number": True}},
            ),
        ),
        selector_policy=policy,
    )
    citation_selection = select_runtime_profile(
        low,
        claims=(
            Claim(
                "A cited claim.",
                claim_id="cite",
                metadata={"features": {"has_citation": True}},
            ),
        ),
        selector_policy=policy.to_dict(),
    )

    assert selection.selected_profile == "latency"
    assert citation_selection.selected_profile == "balanced"
    assert citation_selection.triggered_claim_ids == ("cite",)
    assert policy.to_dict()["sensitive_claim_feature_flags"] == ("has_citation",)
    with pytest.raises(ValueError, match="high_risk_levels"):
        RuntimeProfileSelectorPolicy(high_risk_levels=("bad",))


def test_select_runtime_profile_uses_verification_plan_cost_when_available():
    low = RiskDecision(
        action=ControlAction.ACCEPT,
        risk_level=RiskLevel.LOW,
        confidence=1.0,
        reason="ok",
    )
    balanced_plan = ClaimVerificationPlan(
        run_verifier=True,
        reason="manual",
        verification_scope="all",
        claims=(
            Claim("Current revenue is 10.", claim_id="c1"),
            Claim("Current margin is 20.", claim_id="c2"),
        ),
        verify_claim_ids=("c1", "c2"),
        route_hints=(
            VerificationRouteHint("c1", ("retrieval", "groundedness")),
            VerificationRouteHint("c2", ("retrieval", "groundedness")),
        ),
        retrieval_queries=(
            {"claim_id": "c1", "query": "current revenue"},
            {"claim_id": "c2", "query": "current margin"},
        ),
    )
    audit_plan = ClaimVerificationPlan(
        run_verifier=True,
        reason="manual",
        verification_scope="all",
        claims=(
            Claim("Current revenue is 10.", claim_id="c1"),
            Claim("Current margin is 20.", claim_id="c2"),
            Claim("Current forecast is 30.", claim_id="c3"),
        ),
        verify_claim_ids=("c1", "c2", "c3"),
        route_hints=(
            VerificationRouteHint("c1", ("retrieval", "groundedness")),
            VerificationRouteHint("c2", ("retrieval", "groundedness")),
            VerificationRouteHint("c3", ("retrieval", "groundedness")),
        ),
        retrieval_queries=(
            {"claim_id": "c1", "query": "current revenue"},
            {"claim_id": "c2", "query": "current margin"},
            {"claim_id": "c3", "query": "current forecast"},
        ),
    )

    balanced_selection = select_runtime_profile(low, verification_plan=balanced_plan)
    audit_selection = select_runtime_profile(low, verification_plan=audit_plan)
    disabled_selection = select_runtime_profile(
        low,
        verification_plan=audit_plan,
        selector_policy=RuntimeProfileSelectorPolicy(
            plan_balanced_cost_threshold=None,
            plan_audit_cost_threshold=None,
        ),
    )

    assert balanced_selection.selected_profile == "balanced"
    assert balanced_selection.reason == "verification plan estimated cost requires balanced profile"
    assert balanced_selection.verification_plan_cost["estimated_cost_units"] == pytest.approx(3.5)
    assert audit_selection.selected_profile == "audit"
    assert audit_selection.reason == "verification plan estimated cost requires audit profile"
    assert audit_selection.verification_plan_cost["estimated_cost_units"] == pytest.approx(5.25)
    assert disabled_selection.selected_profile == "latency"
    assert disabled_selection.verification_plan_cost["estimated_cost_units"] == pytest.approx(5.25)


def test_select_pre_generation_profile_routes_prompt_risk():
    low = select_pre_generation_profile("Explain why warmup examples help calibration.")
    current = select_pre_generation_profile("What is the latest price of BTC today?")
    calculation = select_pre_generation_profile("Calculate 42 / 7 and explain the result.")
    domain_state = select_pre_generation_profile(
        "Can this order ship from inventory?",
        metadata={"requires_domain_state": True},
    )

    assert isinstance(low, PreGenerationRiskAssessment)
    assert low.selected_profile == "latency"
    assert low.risk_level == "low"
    assert low.triggered_features == ()
    assert current.selected_profile == "audit"
    assert current.risk_level == "high"
    assert current.triggered_features == ("requires_retrieval", "is_time_sensitive")
    assert current.prompt_features["has_number"] is False
    assert calculation.selected_profile == "balanced"
    assert calculation.risk_level == "medium"
    assert calculation.triggered_features == ("has_number", "has_calculation")
    assert domain_state.selected_profile == "audit"
    assert domain_state.triggered_features == ("requires_domain_state",)
    assert domain_state.triggered_metadata == ("requires_domain_state",)


def test_pre_generation_risk_policy_roundtrip_and_bool_metadata():
    policy = PreGenerationRiskPolicy.from_mapping({
        "high_risk_profile": "balanced",
        "high_risk_metadata_keys": ["needs_audit"],
        "medium_risk_feature_flags": ["has_citation"],
    })

    disabled = select_pre_generation_profile(
        "General explanation with no external facts.",
        metadata={"needs_audit": "false"},
        risk_policy=policy.to_dict(),
    )
    enabled = select_pre_generation_profile(
        "General explanation with no external facts.",
        metadata={"needs_audit": "yes"},
        risk_policy=policy,
    )
    citation = select_pre_generation_profile("Use [1] as the reference.", risk_policy=policy)

    assert disabled.selected_profile == "latency"
    assert disabled.metadata_flags["needs_audit"] is False
    assert enabled.selected_profile == "balanced"
    assert enabled.risk_level == "high"
    assert enabled.triggered_metadata == ("needs_audit",)
    assert citation.selected_profile == "balanced"
    assert citation.risk_level == "medium"
    assert policy.to_dict()["high_risk_metadata_keys"] == ("needs_audit",)
    direct_assessment = PreGenerationRiskAssessment(
        selected_profile="latency",
        risk_level="low",
        reason="direct",
        prompt_features={"has_number": "false", "has_citation": "yes"},
        metadata_flags={"needs_audit": "false", "ambiguous": "maybe"},
    )
    assert direct_assessment.prompt_features == {
        "has_number": False,
        "has_citation": True,
    }
    assert direct_assessment.metadata_flags == {
        "needs_audit": False,
        "ambiguous": True,
    }
    with pytest.raises(ValueError, match="runtime profile selection"):
        PreGenerationRiskPolicy(low_risk_profile="fast")


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


def test_default_correction_policy_plans_action_payloads():
    policy = DefaultCorrectionPolicy()
    claims = extract_claims("Paris is the capital of France. The moon is made of cheese.")
    unsupported_decision = RiskDecision(
        action=ControlAction.RETRIEVE,
        risk_level=RiskLevel.MEDIUM,
        confidence=0.6,
        reason="claim verification found unsupported claim",
    )
    abstain_decision = RiskDecision(
        action=ControlAction.ABSTAIN,
        risk_level=RiskLevel.HIGH,
        confidence=0.9,
        reason="claim verification refuted claim",
    )
    results = (
        VerificationResult(VerificationStatus.SUPPORTED, confidence=0.9, evidence=("atlas",)),
        VerificationResult(VerificationStatus.REFUTED, confidence=0.85, evidence=("nasa",)),
    )

    retrieve = policy.plan(
        unsupported_decision,
        claims=claims,
        verification_results=(results[0], VerificationResult(VerificationStatus.INSUFFICIENT_EVIDENCE, 0.2)),
    )[0]
    abstain = policy.plan(abstain_decision, claims=claims, verification_results=results)[0]

    assert retrieve.action is ControlAction.RETRIEVE
    assert retrieve.payload["retrieval_targets"][0]["claim_id"] == "c2"
    assert retrieve.payload["claim_status_counts"]["insufficient_evidence"] == 1
    assert abstain.action is ControlAction.ABSTAIN
    assert abstain.payload["blocked_claims"][0]["status"] == "refuted"
    assert abstain.payload["blocked_claims"][0]["evidence"] == ("nasa",)


def test_dry_run_action_executor_records_execution_without_side_effects():
    decision = RiskDecision(
        action=ControlAction.ABSTAIN,
        risk_level=RiskLevel.HIGH,
        confidence=0.9,
        reason="claim verification refuted claim",
    )
    claims = extract_claims("The moon is made of cheese.")
    action = DefaultCorrectionPolicy().plan(
        decision,
        claims=claims,
        verification_results=(VerificationResult(VerificationStatus.REFUTED, 0.85, evidence=("nasa",)),),
    )[0]

    result = DryRunActionExecutor().execute(action, context={"request_id": "req-1"})

    assert result.action is ControlAction.ABSTAIN
    assert result.status is ActionExecutionStatus.DRY_RUN
    assert result.output["would_execute"] == "abstain"
    assert result.output["blocked_claims"][0]["claim_id"] == "c1"
    assert result.metadata["side_effects"] is False
    assert result.metadata["context"]["request_id"] == "req-1"


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


def test_claim_coherence_infers_metadata_and_discourse_dependencies():
    claims = (
        Claim("The trial enrolled 100 patients.", claim_id="c1"),
        Claim("Therefore the treatment is proven effective.", claim_id="c2"),
        Claim("The report requires review.", claim_id="c3", metadata={"depends_on": "c2"}),
    )

    dependencies = infer_claim_dependencies(claims)

    assert dependencies == (
        ClaimDependency(
            parent_id="c1",
            child_id="c2",
            relation="discourse_marker",
            source="text_rule",
            reason="claim starts with a discourse marker",
        ),
        ClaimDependency(parent_id="c2", child_id="c3"),
    )


def test_claim_coherence_blocks_supported_child_when_parent_is_missing_or_unsupported():
    claims = (
        Claim("The trial was randomized.", claim_id="c1"),
        Claim("Therefore the treatment is proven effective.", claim_id="c2", metadata={"depends_on": "c1"}),
    )
    results = (
        VerificationResult(VerificationStatus.INSUFFICIENT_EVIDENCE, confidence=0.2),
        VerificationResult(VerificationStatus.SUPPORTED, confidence=0.9, evidence=("abstract",)),
    )

    adjusted, report = apply_claim_coherence(claims, results)

    assert adjusted[0].status is VerificationStatus.INSUFFICIENT_EVIDENCE
    assert adjusted[1].status is VerificationStatus.INSUFFICIENT_EVIDENCE
    assert adjusted[1].confidence == pytest.approx(0.5)
    assert adjusted[1].metadata["claim_coherence"]["blocked"] is True
    assert adjusted[1].metadata["claim_coherence"]["parent_status"] == "insufficient_evidence"
    assert report.blocked_claim_ids == ("c2",)
    assert report.issues[0].parent_id == "c1"

    subset_adjusted, subset_report = apply_claim_coherence(
        claims=(claims[1],),
        verification_results=(VerificationResult(VerificationStatus.SUPPORTED, confidence=0.9),),
        dependency_claims=claims,
    )

    assert subset_adjusted[0].status is VerificationStatus.INSUFFICIENT_EVIDENCE
    assert subset_report.missing_parent_ids == ("c1",)


def test_claim_coherence_propagates_transitive_dependencies_independent_of_order():
    claims = (
        Claim("The trial was randomized.", claim_id="c1"),
        Claim("The treatment caused improvement.", claim_id="c2"),
        Claim("Therefore the treatment should be approved.", claim_id="c3"),
    )
    results = (
        VerificationResult(VerificationStatus.INSUFFICIENT_EVIDENCE, confidence=0.2),
        VerificationResult(VerificationStatus.SUPPORTED, confidence=0.9),
        VerificationResult(VerificationStatus.SUPPORTED, confidence=0.9),
    )
    dependencies = (
        ClaimDependency(parent_id="c2", child_id="c3"),
        ClaimDependency(parent_id="c1", child_id="c2"),
    )

    adjusted, report = apply_claim_coherence(claims, results, dependencies=dependencies)

    assert [result.status for result in adjusted] == [
        VerificationStatus.INSUFFICIENT_EVIDENCE,
        VerificationStatus.INSUFFICIENT_EVIDENCE,
        VerificationStatus.INSUFFICIENT_EVIDENCE,
    ]
    assert report.blocked_claim_ids == ("c2", "c3")
    assert [(issue.parent_id, issue.child_id) for issue in report.issues] == [("c1", "c2"), ("c2", "c3")]


def test_claim_coherence_accepts_json_like_inputs_with_enum_statuses():
    claims = (
        {"text": "The parent claim.", "claim_id": "c1"},
        {
            "text": "The child claim.",
            "claim_id": "c2",
            "metadata": {"dependencies": [{"parent": "c1", "relation": "premise"}]},
        },
    )
    results = (
        {"status": VerificationStatus.REFUTED, "confidence": 0.8, "evidence": None},
        {"status": VerificationStatus.SUPPORTED, "confidence": "0.9", "evidence": ("child evidence",)},
    )

    adjusted, report = apply_claim_coherence(claims, results)

    assert adjusted[0].status is VerificationStatus.REFUTED
    assert adjusted[1].status is VerificationStatus.INSUFFICIENT_EVIDENCE
    assert report.to_dict()["dependencies"][0]["relation"] == "premise"
    assert report.to_dict()["issues"][0]["parent_status"] == "refuted"


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


def test_self_consistency_verifier_supports_claim_with_majority_samples():
    verifier = SelfConsistencyVerifier(
        samples=(
            {"text": "Paris is the capital of France.", "source": "sample-1"},
            "Paris is the capital of France and a major European city.",
            "The capital of France is Paris.",
        ),
        min_overlap=0.55,
        support_threshold=0.60,
    )
    claim = extract_claims("Paris is the capital of France.")[0]

    result = verifier.verify(claim)

    assert result.status is VerificationStatus.SUPPORTED
    assert result.metadata["support_count"] == 3
    assert result.metadata["support_rate"] == pytest.approx(1.0)
    assert result.metadata["decision_rule"] == "support_rate"
    assert result.evidence[0].startswith("sample-1:")


def test_self_consistency_verifier_refutes_numeric_and_negation_disagreement():
    numeric_verifier = SelfConsistencyVerifier(
        samples=(
            "AlphaCorp has 12 offices in Europe.",
            "AlphaCorp has 12 offices in Europe as of 2026.",
            "AlphaCorp has 10 offices in Asia.",
        ),
        min_overlap=0.55,
        refute_threshold=0.50,
    )
    numeric_claim = extract_claims("AlphaCorp has 10 offices in Europe.")[0]

    numeric_result = numeric_verifier.verify(numeric_claim)

    assert numeric_result.status is VerificationStatus.REFUTED
    assert numeric_result.metadata["refute_count"] == 2
    assert numeric_result.metadata["decision_rule"] == "refute_rate"
    assert numeric_result.metadata["sample_decisions"][0]["reason"] == "number_mismatch"

    negation_verifier = SelfConsistencyVerifier(
        samples=("The moon is not made of cheese.", "The moon is not made of cheese."),
        min_overlap=0.50,
    )
    negation_claim = extract_claims("The moon is made of cheese.")[0]

    negation_result = negation_verifier.verify(negation_claim)

    assert negation_result.status is VerificationStatus.REFUTED
    assert negation_result.metadata["sample_decisions"][0]["reason"] == "negation_mismatch"


def test_self_consistency_verifier_uses_context_samples():
    verifier = SelfConsistencyVerifier(samples=(), min_overlap=0.55)
    claim = extract_claims("Water boils at 100 degrees Celsius.")[0]

    missing = verifier.verify(claim)
    result = verifier.verify(
        claim,
        context={
            "selfcheck_samples": (
                "Water boils at 100 degrees Celsius at standard pressure.",
                {"response": "At standard pressure, water boils at 100 degrees Celsius.", "source": "sample-2"},
            )
        },
    )

    assert missing.status is VerificationStatus.NOT_APPLICABLE
    assert missing.metadata["decision_rule"] == "too_few_samples"
    assert result.status is VerificationStatus.SUPPORTED
    assert result.metadata["sample_count"] == 2
    assert result.metadata["sample_decisions"][1]["source"] == "sample-2"


def test_self_consistency_verifier_parses_config_without_bool_numeric_casts():
    verifier = SelfConsistencyVerifier(
        min_samples="2",  # type: ignore[arg-type]
        max_samples="3",  # type: ignore[arg-type]
        min_overlap="0.55",  # type: ignore[arg-type]
        support_threshold="0.60",  # type: ignore[arg-type]
        refute_threshold="0.50",  # type: ignore[arg-type]
        early_stop="false",  # type: ignore[arg-type]
    )

    assert verifier.min_samples == 2
    assert verifier.max_samples == 3
    assert verifier.min_overlap == pytest.approx(0.55)
    assert verifier.support_threshold == pytest.approx(0.60)
    assert verifier.refute_threshold == pytest.approx(0.50)
    assert verifier.early_stop is False
    with pytest.raises(ValueError, match="min_samples"):
        SelfConsistencyVerifier(min_samples=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="min_overlap"):
        SelfConsistencyVerifier(min_overlap=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="support_threshold"):
        SelfConsistencyVerifier(support_threshold=float("nan"))
    with pytest.raises(ValueError, match="early_stop"):
        SelfConsistencyVerifier(early_stop="maybe")  # type: ignore[arg-type]


def test_self_consistency_verifier_early_stops_when_threshold_result_is_fixed():
    verifier = SelfConsistencyVerifier(
        samples=(
            "AlphaCorp has 12 offices in Europe.",
            "AlphaCorp has 12 offices in Europe as of 2026.",
            "AlphaCorp has 10 offices in Europe.",
            "AlphaCorp has 10 offices in Europe.",
            "AlphaCorp has 10 offices in Europe.",
        ),
        min_samples=2,
        min_overlap=0.55,
        refute_threshold=0.40,
        support_threshold=0.80,
        early_stop=True,
    )
    claim = extract_claims("AlphaCorp has 10 offices in Europe.")[0]

    result = verifier.verify(claim)

    assert result.status is VerificationStatus.REFUTED
    assert result.metadata["decision_rule"] == "refute_rate"
    assert result.metadata["early_stop"] is True
    assert result.metadata["early_stop_reason"] == "refute_threshold_guaranteed"
    assert result.metadata["sample_count"] == 5
    assert result.metadata["processed_sample_count"] == 2
    assert result.metadata["skipped_sample_count"] == 3
    assert result.metadata["refute_rate"] == pytest.approx(0.40)
    assert result.metadata["processed_refute_rate"] == pytest.approx(1.0)
    assert len(result.metadata["sample_decisions"]) == 2


def test_self_consistency_sample_budget_status_reports_fixed_threshold_outcome():
    verifier = SelfConsistencyVerifier(
        min_samples=2,
        min_overlap=0.55,
        refute_threshold=0.40,
        support_threshold=0.80,
    )
    claim = extract_claims("AlphaCorp has 10 offices in Europe.")[0]

    status = verifier.sample_budget_status(
        claim,
        (
            "AlphaCorp has 12 offices in Europe.",
            "AlphaCorp has 12 offices in Europe as of 2026.",
        ),
        total_samples=5,
    )

    assert status["can_stop"] is True
    assert status["reason"] == "refute_threshold_guaranteed"
    assert status["sample_count"] == 2
    assert status["remaining_samples"] == 3
    assert status["refute_count"] == 2
    assert status["refute_rate_lower_bound"] == pytest.approx(0.40)


def test_cached_verifier_reuses_identical_claim_context_results():
    class CountingVerifier:
        def __init__(self):
            self.calls = 0

        def verify(self, claim, context=None):
            self.calls += 1
            return VerificationResult(
                VerificationStatus.SUPPORTED,
                confidence=0.9,
                metadata={"claim": claim.text, "context": dict(context or {})},
            )

    base = CountingVerifier()
    verifier = CachedVerifier(base)
    claim = Claim("Inventory is available.", metadata={"state_check": StateCheck("inventory.sku.available")})
    context = {"state": {"inventory": {"sku": {"available": 3}}}}

    first = verifier.verify(claim, context=context)
    second = verifier.verify(claim, context=context)
    changed = verifier.verify(claim, context={"state": {"inventory": {"sku": {"available": 4}}}})

    assert first is second
    assert changed is not first
    assert base.calls == 2
    assert verifier.stats.to_dict()["hits"] == 1
    assert verifier.stats.to_dict()["misses"] == 2


def test_json_trace_cache_roundtrip(tmp_path):
    cache = JsonTraceCache(tmp_path / "trace-cache.json", cache_type="unit_trace")

    stored = cache.put(
        "k1",
        {"results": (VerificationResult(VerificationStatus.SUPPORTED, 0.9),)},
        metadata={"source": "unit"},
    )
    loaded = JsonTraceCache(tmp_path / "trace-cache.json", cache_type="unit_trace").get_record("k1")

    assert stored.key == "k1"
    assert loaded is not None
    assert loaded.payload["results"][0]["status"] == "supported"
    assert loaded.metadata["source"] == "unit"
    assert cache.summary()["records"] == 1
    with pytest.raises(ValueError, match="cache_type"):
        JsonTraceCache(tmp_path / "trace-cache.json", cache_type="other").get_record("k1")


def test_cached_state_source_loads_once_and_copies_state():
    class CountingStateSource:
        def __init__(self):
            self.calls = 0

        def load_state(self):
            self.calls += 1
            return {"inventory": {"sku": {"available": 3}}}

    source = CountingStateSource()
    cached = CachedStateSource(source)

    first = cached.load_state()
    first["inventory"]["sku"]["available"] = 0
    second = cached.load_state()

    assert source.calls == 1
    assert second["inventory"]["sku"]["available"] == 3
    assert cached.stats.to_dict()["hits"] == 1
    assert cached.stats.to_dict()["misses"] == 1


def test_cached_retriever_reuses_query_results():
    retriever = CachedRetriever(InMemoryRetriever(("Paris is the capital of France.",)))
    query = RetrievalQuery(query="Paris capital France", claim_id="c1")

    first = retriever.retrieve(query, limit=1)
    second = retriever.retrieve(query, limit=1)

    assert first == second
    assert second[0].text == "Paris is the capital of France."
    assert retriever.stats.to_dict()["hits"] == 1
    assert retriever.stats.to_dict()["misses"] == 1


def test_in_memory_retriever_preserves_mapping_metadata_fields():
    retriever = InMemoryRetriever((
        {
            "text": "Order R1 is approved for expedited shipping.",
            "source": "shipping:R1",
            "question": "What shipping option is order R1 approved for?",
            "answer": "Order R1 is approved for expedited shipping.",
        },
    ), min_overlap=0.5)

    hits = retriever.retrieve(
        RetrievalQuery(query="What shipping option is order R1 approved for?"),
        limit=1,
    )

    assert hits[0].metadata["question"] == "What shipping option is order R1 approved for?"
    assert hits[0].metadata["answer"] == "Order R1 is approved for expedited shipping."
    assert hits[0].metadata["retriever"] == "InMemoryRetriever"


def test_sqlite_fts_retriever_returns_overlap_hits_or_falls_back():
    retriever = SQLiteFTSRetriever((
        "Paris is the capital of France.",
        "Lyon is a city in France.",
    ), min_overlap=0.6)

    hits = retriever.retrieve(RetrievalQuery(query="Paris capital France"), limit=1)

    assert hits[0].text == "Paris is the capital of France."
    assert hits[0].metadata["token_overlap"] == pytest.approx(1.0)
    if retriever.available:
        assert hits[0].metadata["retriever"] == "SQLiteFTSRetriever"
        assert hits[0].metadata["retriever_backend"] == "sqlite_fts"
    else:
        assert hits[0].metadata["retriever"] == "InMemoryRetriever"
        assert retriever.fallback_reason


def test_sqlite_fts_retriever_can_reuse_persistent_index(tmp_path):
    index_path = tmp_path / "retrieval.sqlite"
    documents = (
        "Paris is the capital of France.",
        "Lyon is a city in France.",
    )
    first = SQLiteFTSRetriever(documents, min_overlap=0.6, index_path=index_path)

    first_hits = first.retrieve(RetrievalQuery(query="Paris capital France"), limit=1)

    assert first_hits[0].text == "Paris is the capital of France."
    assert first.index_reused is False
    if not first.available:
        assert first.fallback_reason
        return

    assert first.index_path == index_path
    assert index_path.exists()
    second = SQLiteFTSRetriever(documents, min_overlap=0.6, index_path=index_path)
    second_hits = second.retrieve(RetrievalQuery(query="Paris capital France"), limit=1)

    assert second.available
    assert second.index_reused is True
    assert second.document_fingerprint == first.document_fingerprint
    assert second_hits[0].text == "Paris is the capital of France."

    changed = SQLiteFTSRetriever((*documents, "Marseille is a port city."), min_overlap=0.6, index_path=index_path)
    assert changed.available
    assert changed.index_reused is False
    assert changed.document_fingerprint != first.document_fingerprint


def test_question_answer_verifier_checks_structured_question_answers():
    verifier = QuestionAnswerVerifier([
        QuestionAnswerFact(
            question="What is the capital of France?",
            answer="Paris",
            source="qa:facts",
        )
    ])

    supported = verifier.verify(
        Claim("Paris", metadata={"question": "What is the capital of France?", "answer": "Paris"})
    )
    refuted = verifier.verify(
        Claim("Lyon"),
        context={"statement": {"question": "What is the capital of France?", "answer": "Lyon"}},
    )
    unknown = verifier.verify(
        Claim("Madrid"),
        context={"statement": {"question": "What is the capital of Spain?", "answer": "Madrid"}},
    )

    assert supported.status is VerificationStatus.SUPPORTED
    assert supported.metadata["decision_rule"] == "answer_match"
    assert refuted.status is VerificationStatus.REFUTED
    assert refuted.metadata["decision_rule"] == "answer_mismatch"
    assert "Paris" in refuted.evidence[0]
    assert unknown.status is VerificationStatus.INSUFFICIENT_EVIDENCE
    assert unknown.metadata["decision_rule"] == "question_not_found"


def test_calculator_verifier_supports_and_refutes_arithmetic_claims():
    verifier = CalculatorVerifier()

    supported = verifier.verify(Claim("2 + 2 = 4."))
    refuted = verifier.verify(Claim("2 + 2 = 5."))
    extracted = verifier.verify(extract_claims("2 plus 2 is 5.")[0])
    structured = verifier.verify(
        Claim("The computed total is 12.", metadata={"calculation": {"expression": "3 * 4", "expected": 12}})
    )

    assert supported.status is VerificationStatus.SUPPORTED
    assert supported.metadata["decision_rule"] == "calculation_match"
    assert refuted.status is VerificationStatus.REFUTED
    assert refuted.metadata["decision_rule"] == "calculation_mismatch"
    assert refuted.metadata["actual"] == pytest.approx(4.0)
    assert extracted.status is VerificationStatus.REFUTED
    assert extracted.metadata["expression"] == "2 + 2"
    assert structured.status is VerificationStatus.SUPPORTED
    assert structured.evidence[0].startswith("calculator: 3 * 4 = 12")


def test_calculator_verifier_handles_non_applicable_and_unsafe_expressions():
    verifier = CalculatorVerifier()

    not_applicable = verifier.verify(Claim("Paris is the capital of France."))
    unsafe = verifier.verify(
        Claim("Bad calculation.", metadata={"expression": "__import__('os').system('true')", "expected": 0})
    )
    divided_by_zero = verifier.verify(Claim("1 / 0 = 0."))

    assert not_applicable.status is VerificationStatus.NOT_APPLICABLE
    assert not_applicable.metadata["decision_rule"] == "no_calculation"
    assert unsafe.status is VerificationStatus.ERROR
    assert unsafe.metadata["decision_rule"] == "calculation_error"
    assert divided_by_zero.status is VerificationStatus.ERROR
    assert divided_by_zero.explanation == "division by zero"


def test_calculator_verifier_rejects_non_finite_tolerance_and_expected_values():
    verifier = CalculatorVerifier()

    infinite_tolerance = verifier.verify(
        Claim(
            "Bad calculation.",
            metadata={
                "calculation": {
                    "expression": "2 + 2",
                    "expected": 5,
                    "tolerance": "inf",
                },
            },
        )
    )
    infinite_expected = verifier.verify(
        Claim(
            "Bad calculation.",
            metadata={"calculation": {"expression": "2 + 2", "expected": "nan"}},
        )
    )

    assert infinite_tolerance.status is VerificationStatus.ERROR
    assert infinite_tolerance.metadata["decision_rule"] == "invalid_calculation_config"
    assert infinite_expected.status is VerificationStatus.ERROR
    assert infinite_expected.metadata["decision_rule"] == "invalid_calculation_config"
    with pytest.raises(ValueError, match="default_tolerance"):
        CalculatorVerifier(default_tolerance=float("inf"))
    with pytest.raises(ValueError, match="max_abs_value"):
        CalculatorVerifier(max_abs_value=float("nan"))


def test_structured_state_verifier_supports_and_refutes_business_rules():
    verifier = StructuredStateVerifier(
        state={
            "inventory": {"sku_123": {"available": 12}},
            "account": {"status": "active", "tier": "enterprise"},
        }
    )

    supported = verifier.verify(
        Claim(
            "SKU 123 has enough available inventory.",
            metadata={"state_check": {"path": "inventory.sku_123.available", "operator": ">=", "value": 10}},
        )
    )
    refuted = verifier.verify(
        Claim(
            "Account is suspended.",
            metadata={"state_check": {"path": "account.status", "operator": "eq", "value": "suspended"}},
        )
    )
    membership = verifier.verify(
        Claim(
            "Account tier is allowed.",
            metadata={"state_check": {"path": "account.tier", "operator": "in", "value": ["pro", "enterprise"]}},
        )
    )

    assert supported.status is VerificationStatus.SUPPORTED
    assert supported.metadata["decision_rule"] == "state_check_passed"
    assert supported.metadata["actual"] == 12
    assert refuted.status is VerificationStatus.REFUTED
    assert refuted.metadata["decision_rule"] == "state_check_failed"
    assert "suspended" in refuted.evidence[0]
    assert membership.status is VerificationStatus.SUPPORTED


def test_structured_state_verifier_uses_context_state_and_claim_specific_checks():
    verifier = StructuredStateVerifier(state={"inventory": {"sku_123": {"available": 2}}})
    claim = Claim("Inventory is updated.", claim_id="c1")

    result = verifier.verify(
        claim,
        context={
            "state": {"inventory": {"sku_123": {"available": 5}}},
            "state_checks": {
                "c1": StateCheck(path="inventory.sku_123.available", operator="between", value=[3, 6])
            },
        },
    )

    assert result.status is VerificationStatus.SUPPORTED
    assert result.metadata["actual"] == 5
    assert result.metadata["operator"] == "between"


def test_structured_state_verifier_reports_missing_and_invalid_checks():
    verifier = StructuredStateVerifier(state={"account": {"balance": 10}})

    not_applicable = verifier.verify(Claim("No structured state check."))
    missing = verifier.verify(
        Claim("Missing state.", metadata={"state_check": {"path": "account.limit", "operator": "exists"}})
    )
    invalid = verifier.verify(
        Claim(
            "Invalid numeric check.",
            metadata={"state_check": {"path": "account.balance", "operator": ">", "value": "x"}},
        )
    )

    assert not_applicable.status is VerificationStatus.NOT_APPLICABLE
    assert not_applicable.metadata["decision_rule"] == "no_state_check"
    assert missing.status is VerificationStatus.INSUFFICIENT_EVIDENCE
    assert missing.metadata["decision_rule"] == "state_path_missing"
    assert invalid.status is VerificationStatus.ERROR
    assert invalid.metadata["decision_rule"] == "evaluation_error"


def test_sqlite_state_source_loads_database_state_for_verifier(tmp_path):
    db_path = tmp_path / "state.db"
    connection = sqlite3.connect(db_path)
    connection.execute("create table inventory (sku text primary key, available integer)")
    connection.execute("create table accounts (id text primary key, status text, tier text)")
    connection.execute("insert into inventory values (?, ?)", ("sku_123", 12))
    connection.execute("insert into accounts values (?, ?, ?)", ("acct_1", "active", "enterprise"))
    connection.commit()
    connection.close()

    source = SQLiteStateSource(
        db_path,
        queries=(
            SQLiteStateQuery(
                path="inventory.sku_123.available",
                sql="select available from inventory where sku = ?",
                params=("sku_123",),
            ),
            {
                "path": "account.status",
                "sql": "select status from accounts where id = ?",
                "params": ("acct_1",),
                "column": "status",
            },
            {
                "path": "account.profile",
                "sql": "select status, tier from accounts where id = ?",
                "params": ("acct_1",),
            },
        ),
    )

    state = source.load_state()
    verifier = StructuredStateVerifier.from_source(source)
    supported = verifier.verify(
        Claim(
            "SKU 123 has enough inventory.",
            metadata={"state_check": {"path": "inventory.sku_123.available", "operator": ">=", "value": 10}},
        )
    )
    refuted = verifier.verify(
        Claim("Account is suspended.", metadata={"state_check": {"path": "account.status", "value": "suspended"}})
    )

    assert state["inventory"]["sku_123"]["available"] == 12
    assert state["account"]["status"] == "active"
    assert state["account"]["profile"] == {"status": "active", "tier": "enterprise"}
    assert supported.status is VerificationStatus.SUPPORTED
    assert refuted.status is VerificationStatus.REFUTED
    assert "sqlite" not in supported.metadata


def test_sqlite_state_source_handles_missing_required_queries(tmp_path):
    db_path = tmp_path / "state.db"
    connection = sqlite3.connect(db_path)
    connection.execute("create table inventory (sku text primary key, available integer)")
    connection.commit()
    connection.close()

    optional = SQLiteStateSource(
        db_path,
        queries=(
            {
                "path": "inventory.missing.available",
                "sql": "select available from inventory where sku = ?",
                "params": ("missing",),
                "required": "false",
            },
        ),
    )
    required = SQLiteStateSource(
        db_path,
        queries=(
            {
                "path": "inventory.missing.available",
                "sql": "select available from inventory where sku = ?",
                "params": ("missing",),
                "required": True,
            },
        ),
    )

    assert optional.load_state() == {}
    with pytest.raises(ValueError, match="returned no rows"):
        required.load_state()
    with pytest.raises(ValueError, match="required must be"):
        SQLiteStateQuery.from_mapping({"path": "x", "sql": "select 1", "required": "maybe"})


def test_tool_output_state_source_maps_action_results_for_state_verifier():
    class ReservationToolExecutor:
        def execute(self, request, context=None):
            return ActionResult(
                action=request.action,
                status=ActionExecutionStatus.SUCCEEDED,
                request_id=request.request_id,
                output={
                    "reservation": {
                        "order_id": "ord_1",
                        "sku": "sku_123",
                        "status": "reserved",
                        "remaining_available": 7,
                    }
                },
                metadata={"executor": type(self).__name__, "context": dict(context or {})},
            )

        def execute_many(self, requests, context=None):
            return tuple(self.execute(request, context=context) for request in requests)

    request = ActionRequest(
        action=ControlAction.RETRIEVE,
        reason="check reservation tool output",
        request_id="reserve-1",
    )
    registry = ActionExecutorRegistry().register(ControlAction.RETRIEVE, ReservationToolExecutor())
    action_result = registry.execute(request, context={"request_id": "req-tool-output"})
    source = ToolOutputStateSource(
        action_results=(action_result,),
        mappings=(
            ToolOutputMapping(
                state_path="inventory.sku_123.available",
                output_path="reservation.remaining_available",
                action=ControlAction.RETRIEVE,
                request_id="reserve-1",
                required=True,
            ),
            {
                "state_path": "orders.ord_1.status",
                "output_path": "reservation.status",
                "action": "retrieve",
                "request_id": "reserve-1",
            },
        ),
    )

    state = source.load_state()
    verifier = StructuredStateVerifier.from_source(source)
    supported = verifier.verify(
        Claim(
            "Reservation tool left 7 units available.",
            metadata={
                "state_check": {
                    "path": "inventory.sku_123.available",
                    "operator": "eq",
                    "value": 7,
                    "source": "reservation_tool_output",
                }
            },
        )
    )
    refuted = verifier.verify(
        Claim(
            "Reservation tool left 9 units available.",
            metadata={"state_check": {"path": "inventory.sku_123.available", "operator": "eq", "value": 9}},
        )
    )

    assert state["inventory"]["sku_123"]["available"] == 7
    assert state["orders"]["ord_1"]["status"] == "reserved"
    assert state["tool_outputs"]["by_request_id"]["reserve-1"]["reservation"]["status"] == "reserved"
    assert state["tool_outputs"]["first_by_action"]["retrieve"]["reservation"]["remaining_available"] == 7
    assert supported.status is VerificationStatus.SUPPORTED
    assert supported.metadata["actual"] == 7
    assert refuted.status is VerificationStatus.REFUTED


def test_tool_output_state_source_handles_raw_outputs_defaults_and_required_mappings():
    source = ToolOutputStateSource(
        outputs={"calculator": {"result": 42}},
        mappings=(
            {"state_path": "answers.value", "output_path": "calculator.result", "required": "true"},
            {"state_path": "answers.units", "output_path": "calculator.units", "default": "items"},
        ),
    )
    required_missing = ToolOutputStateSource(
        outputs={"calculator": {"result": 42}},
        mappings=(
            {"state_path": "answers.value", "output_path": "calculator.missing", "required": True},
        ),
    )

    state = source.load_state()

    assert state["answers"] == {"value": 42, "units": "items"}
    assert state["tool_outputs"]["input"]["calculator"]["result"] == 42
    with pytest.raises(ValueError, match="required tool output mapping"):
        required_missing.load_state()


def test_tool_output_state_source_ignores_failed_action_outputs_for_state_mapping():
    failed_result = ActionResult(
        action=ControlAction.RETRIEVE,
        status=ActionExecutionStatus.FAILED,
        request_id="retrieve-failed",
        output={"reservation": {"remaining_available": 0}},
        error="tool failed",
    )
    timed_out_result = ActionResult(
        action=ControlAction.RETRIEVE,
        status=ActionExecutionStatus.TIMED_OUT,
        request_id="retrieve-timeout",
        output={"reservation": {"remaining_available": 1}},
        error="tool timed out",
    )
    source = ToolOutputStateSource(
        action_results=(failed_result, timed_out_result),
        mappings=(
            ToolOutputMapping(
                state_path="inventory.sku_123.available",
                output_path="reservation.remaining_available",
                action=ControlAction.RETRIEVE,
                request_id="retrieve-failed",
                required=False,
            ),
        ),
    )
    required_source = ToolOutputStateSource(
        action_results=(failed_result,),
        mappings=(
            ToolOutputMapping(
                state_path="inventory.sku_123.available",
                output_path="reservation.remaining_available",
                action=ControlAction.RETRIEVE,
                request_id="retrieve-failed",
                required=True,
            ),
        ),
    )

    state = source.load_state()

    assert "inventory" not in state
    assert state["tool_outputs"]["results"][0]["status"] == "failed"
    assert "by_request_id" not in state["tool_outputs"]
    with pytest.raises(ValueError, match="required tool output mapping"):
        required_source.load_state()


def test_composite_verifier_skips_not_applicable_tool_results():
    fallback = InMemoryVerifier({normalize_claim_text("Paris is the capital of France"): VerificationStatus.SUPPORTED})
    verifier = CompositeVerifier((CalculatorVerifier(), fallback))

    arithmetic = verifier.verify(Claim("2 + 2 = 5."))
    factual = verifier.verify(Claim("Paris is the capital of France."))

    assert arithmetic.status is VerificationStatus.REFUTED
    assert arithmetic.metadata["selected_verifier"] == "CalculatorVerifier"
    assert factual.status is VerificationStatus.SUPPORTED
    assert factual.metadata["selected_verifier"] == "InMemoryVerifier"
    assert factual.metadata["skipped_verifiers"][0]["verifier"] == "CalculatorVerifier"


def test_routed_verifier_selects_routes_from_metadata_context_and_text():
    fallback = InMemoryVerifier({normalize_claim_text("Paris is the capital of France"): VerificationStatus.SUPPORTED})
    verifier = RoutedVerifier((
        VerifierRoute(
            "calculator",
            CalculatorVerifier(),
            metadata_keys=("calculation", "expression"),
            context_keys=("calculation", "expression"),
            text_patterns=(r"\d\s*[+*/-]\s*\d\s*=",),
        ),
        VerifierRoute("fallback", fallback, fallback=True),
    ))

    text_route = verifier.verify(Claim("2 + 2 = 5."))
    context_route = verifier.verify(
        Claim("The computed total is wrong."),
        context={"calculation": {"expression": "6 / 3", "expected": 3}},
    )
    fallback_route = verifier.verify(Claim("Paris is the capital of France."))

    assert text_route.status is VerificationStatus.REFUTED
    assert text_route.metadata["selected_route"] == "calculator"
    assert "text_pattern:" in text_route.metadata["matched_route_details"][0]["match_reasons"][0]
    assert context_route.status is VerificationStatus.REFUTED
    assert context_route.metadata["selected_route"] == "calculator"
    assert context_route.metadata["matched_route_details"][0]["match_reasons"] == ("context:calculation",)
    assert fallback_route.status is VerificationStatus.SUPPORTED
    assert fallback_route.metadata["selected_route"] == "fallback"
    assert fallback_route.metadata["matched_route_details"][0]["match_reasons"] == ("fallback",)
    for routed in (text_route, context_route, fallback_route):
        assert routed.metadata["attempted_route_count"] == 1.0
        assert math.isfinite(routed.metadata["total_duration_seconds"])
        assert math.isfinite(routed.metadata["selected_route_duration_seconds"])
        assert routed.metadata["total_duration_seconds"] >= routed.metadata["selected_route_duration_seconds"] >= 0.0


def test_routed_verifier_uses_extracted_calculation_metadata_without_text_pattern():
    fallback = InMemoryVerifier({})
    verifier = RoutedVerifier((
        VerifierRoute("calculator", CalculatorVerifier(), metadata_keys=("calculation",)),
        VerifierRoute("fallback", fallback, fallback=True),
    ))

    result = verifier.verify(extract_claims("2 plus 2 is 5.")[0])

    assert result.status is VerificationStatus.REFUTED
    assert result.metadata["selected_route"] == "calculator"
    assert result.metadata["matched_route_details"][0]["match_reasons"] == ("metadata:calculation",)
    assert result.metadata["expression"] == "2 + 2"


def test_routed_verifier_parses_string_feature_flags():
    verifier = RoutedVerifier((
        VerifierRoute("calculator", CalculatorVerifier(), feature_flags=("has_calculation",)),
    ))

    matched = verifier.verify(
        Claim(
            "The computed answer is wrong.",
            metadata={
                "features": {"has_calculation": "true"},
                "calculation": {"expression": "2 + 2", "expected": 5},
            },
        )
    )
    skipped = verifier.verify(
        Claim(
            "The computed answer is wrong.",
            metadata={
                "features": {"has_calculation": "false"},
                "calculation": {"expression": "2 + 2", "expected": 5},
            },
        )
    )

    assert matched.status is VerificationStatus.REFUTED
    assert matched.metadata["selected_route"] == "calculator"
    assert matched.metadata["matched_route_details"][0]["match_reasons"] == (
        "feature_flag:has_calculation",
    )
    assert skipped.status is VerificationStatus.NOT_APPLICABLE
    assert skipped.metadata["matched_routes"] == ()


def test_routed_verifier_can_prioritize_structured_state_adapter():
    fallback = InMemoryVerifier({normalize_claim_text("Fallback fact"): VerificationStatus.SUPPORTED})
    verifier = RoutedVerifier((
        VerifierRoute(
            "state",
            StructuredStateVerifier(state={"quota": {"remaining": 0}}),
            metadata_keys=("state_check",),
        ),
        VerifierRoute("fallback", fallback, fallback=True),
    ))

    state_route = verifier.verify(
        Claim("Quota remains.", metadata={"state_check": {"path": "quota.remaining", "operator": ">", "value": 0}})
    )
    fallback_route = verifier.verify(Claim("Fallback fact."))

    assert state_route.status is VerificationStatus.REFUTED
    assert state_route.metadata["selected_route"] == "state"
    assert state_route.metadata["selected_verifier"] == "StructuredStateVerifier"
    assert fallback_route.status is VerificationStatus.SUPPORTED
    assert fallback_route.metadata["selected_route"] == "fallback"


def test_routed_verifier_reports_not_applicable_when_no_route_matches():
    verifier = RoutedVerifier((
        VerifierRoute("calculator", CalculatorVerifier(), metadata_keys=("calculation",)),
    ))

    result = verifier.verify(Claim("Paris is the capital of France."))

    assert result.status is VerificationStatus.NOT_APPLICABLE
    assert result.metadata["matched_routes"] == ()


def test_routed_verifier_can_fall_through_on_insufficient_evidence():
    qa = QuestionAnswerVerifier([QuestionAnswerFact(question="Q?", answer="A")])
    fallback = InMemoryVerifier({normalize_claim_text("Fallback fact"): VerificationStatus.SUPPORTED})
    verifier = RoutedVerifier((
        VerifierRoute(
            "structured_qa",
            qa,
            context_keys=("statement.question", "statement.answer"),
            fallthrough_statuses=(VerificationStatus.INSUFFICIENT_EVIDENCE,),
        ),
        VerifierRoute("fallback", fallback, fallback=True),
    ))

    result = verifier.verify(
        Claim("Fallback fact."),
        context={"statement": {"question": "Unknown?", "answer": "A"}},
    )

    assert result.status is VerificationStatus.SUPPORTED
    assert result.metadata["selected_route"] == "fallback"
    assert result.metadata["skipped_routes"][0]["route"] == "structured_qa"
    assert result.metadata["skipped_routes"][0]["match_reasons"] == (
        "context:statement.question",
        "context:statement.answer",
    )
    assert result.metadata["skipped_routes"][0]["status"] == "insufficient_evidence"
    assert result.metadata["attempted_route_count"] == 2.0
    assert math.isfinite(result.metadata["total_duration_seconds"])
    assert math.isfinite(result.metadata["selected_route_duration_seconds"])
    assert math.isfinite(result.metadata["skipped_routes"][0]["duration_seconds"])
    assert result.metadata["total_duration_seconds"] >= result.metadata["selected_route_duration_seconds"] >= 0.0


def test_routed_verifier_can_cap_route_fanout():
    qa = QuestionAnswerVerifier([QuestionAnswerFact(question="Q?", answer="A")])
    fallback = InMemoryVerifier({normalize_claim_text("Fallback fact"): VerificationStatus.SUPPORTED})
    verifier = RoutedVerifier(
        (
            VerifierRoute(
                "structured_qa",
                qa,
                context_keys=("statement.question", "statement.answer"),
                fallthrough_statuses=(VerificationStatus.INSUFFICIENT_EVIDENCE,),
            ),
            VerifierRoute("fallback", fallback, fallback=True),
        ),
        max_attempted_routes=1,
    )

    result = verifier.verify(
        Claim("Fallback fact."),
        context={"statement": {"question": "Unknown?", "answer": "A"}},
    )

    assert result.status is VerificationStatus.INSUFFICIENT_EVIDENCE
    assert result.metadata["selected_route"] == "structured_qa"
    assert result.metadata["attempted_route_count"] == 1.0
    assert result.metadata["route_budget_limit"] == 1
    assert result.metadata["route_budget_exhausted"] is True
    assert result.metadata["route_stop_reason"] == "max_attempted_routes"
    assert result.metadata["selected_route_was_fallthrough"] is True
    assert result.metadata["unattempted_routes"] == ("fallback",)
    assert result.metadata["skipped_routes"] == ()


def test_routed_verifier_fails_closed_when_not_applicable_route_exhausts_budget():
    fallback = InMemoryVerifier({normalize_claim_text("Fallback fact"): VerificationStatus.SUPPORTED})
    verifier = RoutedVerifier(
        (
            VerifierRoute("calculator", CalculatorVerifier(), fallback=True),
            VerifierRoute("fallback", fallback, fallback=True),
        ),
        max_attempted_routes=1,
    )

    result = verifier.verify(Claim("Fallback fact."))

    assert result.status is VerificationStatus.INSUFFICIENT_EVIDENCE
    assert result.metadata["budget_exhaustion_original_status"] == "not_applicable"
    assert result.metadata["route_budget_exhausted"] is True
    assert result.metadata["unattempted_routes"] == ("fallback",)

    artifact = CalibrationArtifact(
        model_id="tiny",
        target_layer=-1,
        scores=(CalibrationScore("maha_last", threshold=1.0),),
        eigentruth_version="0.1.0",
    )
    decision = RiskController(artifact).decide(
        {"maha_last": 0.0},
        verification_results=(result,),
    )

    assert decision.action is ControlAction.RETRIEVE
    assert decision.risk_level is RiskLevel.MEDIUM


def test_routed_verifier_rejects_invalid_route_fanout_budget():
    route = VerifierRoute("calculator", CalculatorVerifier(), metadata_keys=("calculation",))

    with pytest.raises(ValueError, match="max_attempted_routes"):
        RoutedVerifier((route,), max_attempted_routes=0)
    with pytest.raises(ValueError, match="max_attempted_routes"):
        RoutedVerifier((route,), max_attempted_routes=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="max_attempted_routes"):
        RoutedVerifier((route,), max_attempted_routes=1.5)  # type: ignore[arg-type]


def test_in_memory_world_model_adapter_verifies_and_predicts_state():
    verifier = InMemoryVerifier({normalize_claim_text("Inventory is 10"): VerificationStatus.SUPPORTED})
    adapter = InMemoryWorldModelAdapter(verifier=verifier)
    claim = extract_claims("Inventory is 10.")[0]

    result = adapter.verify(claim)
    prediction = adapter.predict({"inventory": 10}, {"set": {"inventory": 8}})
    nested_prediction = adapter.predict(
        {"inventory": {"sku_123": {"available": 10}}, "quota": {"used": 1}},
        {
            "decrement": {"inventory.sku_123.available": 3},
            "increment": {"quota.used": 2},
        },
    )

    assert result.status is VerificationStatus.SUPPORTED
    assert prediction.state["inventory"] == 8
    assert nested_prediction.state["inventory"]["sku_123"]["available"] == 7
    assert nested_prediction.state["quota"]["used"] == 3
    assert "Inventory" in adapter.explain(claim)


def test_state_transition_verifier_checks_predicted_postconditions():
    adapter = InMemoryWorldModelAdapter(verifier=InMemoryVerifier({}))
    verifier = StateTransitionVerifier(
        world_model=adapter,
        state={"inventory": {"sku_123": {"available": 10}}, "orders": {"ord_1": {"status": "pending"}}},
    )
    supported = verifier.verify(
        Claim(
            "Shipping the order leaves 7 units available.",
            metadata={
                "state_transition": StateTransitionCheck(
                    action={
                        "decrement": {"inventory.sku_123.available": 3},
                        "set": {"orders.ord_1.status": "shipped"},
                    },
                    postcondition={"path": "inventory.sku_123.available", "operator": "eq", "value": 7},
                    source="order_world_model",
                )
            },
        )
    )
    refuted = verifier.verify(
        Claim(
            "Shipping the order leaves 10 units available.",
            metadata={
                "state_transition": {
                    "action": {"decrement": {"inventory.sku_123.available": 3}},
                    "postcondition": {"path": "inventory.sku_123.available", "operator": "eq", "value": 10},
                }
            },
        )
    )

    assert supported.status is VerificationStatus.SUPPORTED
    assert supported.metadata["verifier"] == "state_transition"
    assert supported.metadata["decision_rule"] == "transition_postcondition_passed"
    assert supported.metadata["world_model"] == "InMemoryWorldModelAdapter"
    assert supported.metadata["actual"] == 7
    assert refuted.status is VerificationStatus.REFUTED
    assert refuted.metadata["decision_rule"] == "transition_postcondition_failed"
    assert verifier.state["inventory"]["sku_123"]["available"] == 10


def test_state_transition_verifier_uses_context_state_and_claim_specific_transition():
    adapter = InMemoryWorldModelAdapter(verifier=InMemoryVerifier({}))
    verifier = StateTransitionVerifier(world_model=adapter, state={"quota": {"remaining": 5}})
    result = verifier.verify(
        Claim("The request consumes quota.", claim_id="c1"),
        context={
            "state": {"quota": {"remaining": 4}},
            "state_transitions": {
                "c1": {
                    "action": {"set": {"quota.remaining": 2}},
                    "state_check": {"path": "quota.remaining", "operator": "eq", "value": 2},
                }
            },
        },
    )
    not_applicable = verifier.verify(Claim("No transition metadata."))
    invalid = verifier.verify(
        Claim("Invalid transition.", metadata={"state_transition": {"postcondition": {"path": "x"}}})
    )

    assert result.status is VerificationStatus.SUPPORTED
    assert result.metadata["actual"] == 2
    assert not_applicable.status is VerificationStatus.NOT_APPLICABLE
    assert not_applicable.metadata["decision_rule"] == "no_state_transition"
    assert invalid.status is VerificationStatus.ERROR
    assert invalid.metadata["decision_rule"] == "invalid_state_transition"


def test_action_executor_registry_uses_fallback_and_registered_executor():
    fallback_request = ActionRequest(
        action=ControlAction.ABSTAIN,
        reason="refuted claim",
        payload={"message": "blocked"},
    )
    registry = ActionExecutorRegistry()

    fallback_result = registry.execute(fallback_request)

    assert fallback_result.status is ActionExecutionStatus.DRY_RUN
    assert fallback_result.output["would_execute"] == "abstain"

    retrieve_request = ActionRequest(
        action=ControlAction.RETRIEVE,
        reason="unsupported claim",
        payload={"retrieval_targets": ({"claim_id": "c1", "text": "Paris capital France"},)},
    )
    registry.register(
        ControlAction.RETRIEVE,
        RetrievalActionExecutor(InMemoryRetriever(("Paris is the capital of France.",))),
    )

    retrieve_result = registry.execute(retrieve_request)

    assert retrieve_result.status is ActionExecutionStatus.SUCCEEDED
    assert retrieve_result.output["queries"][0]["claim_id"] == "c1"
    assert retrieve_result.output["hits"][0]["text"] == "Paris is the capital of France."
    assert retrieve_result.metadata["executor"] == "RetrievalActionExecutor"


def test_action_executor_registry_converts_executor_exception_to_failed_result():
    class ExplodingExecutor:
        def execute(self, request, context=None):
            raise RuntimeError("boom")

        def execute_many(self, requests, context=None):
            return tuple(self.execute(request, context=context) for request in requests)

    request = ActionRequest(action=ControlAction.RETRIEVE, reason="unsupported claim", request_id="req-explode")
    registry = ActionExecutorRegistry({ControlAction.RETRIEVE: ExplodingExecutor()})

    result = registry.execute(request, context={"request_id": "req-explode"})

    assert result.status is ActionExecutionStatus.FAILED
    assert result.request_id == "req-explode"
    assert result.metadata["executor"] == "ActionExecutorRegistry"
    assert result.metadata["wrapped_executor"] == "ExplodingExecutor"
    assert result.metadata["possible_side_effects"] is True
    assert "boom" in result.error


def test_policy_guarded_action_executor_validates_side_effect_contract():
    class RecordingExecutor:
        def __init__(self):
            self.calls = 0

        def execute(self, request, context=None):
            self.calls += 1
            return ActionResult(
                action=request.action,
                status=ActionExecutionStatus.SUCCEEDED,
                output={"ok": True},
                metadata={"executor": type(self).__name__, "context": dict(context or {}), "side_effects": True},
                request_id=request.request_id,
            )

        def execute_many(self, requests, context=None):
            return tuple(self.execute(request, context=context) for request in requests)

    wrapped = RecordingExecutor()
    ledger = InMemoryActionExecutionLedger()
    executor = PolicyGuardedActionExecutor(
        wrapped,
        policy=ActionExecutionPolicy(
            side_effecting=True,
            require_request_id=True,
            require_idempotency_key=True,
            max_timeout_seconds=5.0,
        ),
        idempotency_ledger=ledger,
    )
    missing_key = ActionRequest(
        action=ControlAction.EXECUTE_TOOL,
        reason="reserve inventory",
        request_id="reserve-1",
    )
    blocked = executor.execute(missing_key)
    too_slow = executor.execute(
        ActionRequest(
            action=ControlAction.EXECUTE_TOOL,
            reason="reserve inventory",
            metadata={"idempotency_key": "reserve-1", "timeout_seconds": 30.0},
            request_id="reserve-1",
        )
    )
    allowed = executor.execute(
        ActionRequest(
            action=ControlAction.EXECUTE_TOOL,
            reason="reserve inventory",
            metadata={"idempotency_key": "reserve-1", "timeout_seconds": 3.0},
            request_id="reserve-1",
        ),
        context={"request_id": "req-1"},
    )
    replayed = executor.execute(
        ActionRequest(
            action=ControlAction.EXECUTE_TOOL,
            reason="reserve inventory",
            metadata={"idempotency_key": "reserve-1", "timeout_seconds": 3.0},
            request_id="reserve-1",
        ),
        context={"request_id": "req-1"},
    )
    replay_mismatch = executor.execute(
        ActionRequest(
            action=ControlAction.EXECUTE_TOOL,
            reason="reserve inventory",
            metadata={"idempotency_key": "reserve-1", "timeout_seconds": 3.0},
            request_id="reserve-1",
        ),
        context={"request_id": "req-2"},
    )

    assert blocked.status is ActionExecutionStatus.FAILED
    assert "idempotency_key is required" in blocked.error
    assert blocked.metadata["side_effects"] is False
    assert too_slow.status is ActionExecutionStatus.FAILED
    assert "timeout_seconds exceeds max_timeout_seconds" in too_slow.error
    assert wrapped.calls == 1
    assert allowed.status is ActionExecutionStatus.SUCCEEDED
    assert allowed.metadata["policy_guard"] == "PolicyGuardedActionExecutor"
    assert allowed.metadata["idempotency_key"] == "reserve-1"
    assert allowed.metadata["idempotency_request_fingerprint"]
    assert allowed.metadata["timeout_seconds"] == 3.0
    assert allowed.metadata["timeout_enforced"] is False
    assert allowed.metadata["idempotency_replayed"] is False
    assert replayed.status is ActionExecutionStatus.SUCCEEDED
    assert replayed.output == {"ok": True}
    assert replayed.metadata["idempotency_replayed"] is True
    assert replayed.metadata["idempotency_request_fingerprint"] == allowed.metadata["idempotency_request_fingerprint"]
    assert replayed.metadata["side_effects"] is False
    assert replayed.metadata["original_side_effects"] is True
    assert replay_mismatch.status is ActionExecutionStatus.FAILED
    assert replay_mismatch.metadata["idempotency_replay_blocked"] is True
    assert replay_mismatch.metadata["idempotency_replayed"] is False
    assert "fingerprint" in replay_mismatch.error
    assert wrapped.calls == 1
    assert ledger.get("reserve-1") == allowed


def test_policy_guarded_action_executor_fails_closed_on_executor_and_ledger_errors():
    class ExplodingExecutor:
        def execute(self, request, context=None):
            raise RuntimeError("executor boom")

    class ExplodingLedger:
        def __init__(self, *, fail_get=False, fail_record=False):
            self.fail_get = fail_get
            self.fail_record = fail_record

        def get(self, key):
            if self.fail_get:
                raise RuntimeError("ledger get boom")
            return None

        def record(self, key, result):
            if self.fail_record:
                raise RuntimeError("ledger record boom")

    class SuccessfulExecutor:
        def execute(self, request, context=None):
            return ActionResult(
                action=request.action,
                status=ActionExecutionStatus.SUCCEEDED,
                output={"ok": True},
                metadata={"side_effects": True},
                request_id=request.request_id,
            )

    request = ActionRequest(
        action=ControlAction.EXECUTE_TOOL,
        reason="reserve inventory",
        metadata={"idempotency_key": "reserve-2"},
        request_id="reserve-2",
    )
    policy = ActionExecutionPolicy(side_effecting=True, require_idempotency_key=True)

    missing_ledger = PolicyGuardedActionExecutor(SuccessfulExecutor(), policy=policy).execute(request)
    wrapped_failure = PolicyGuardedActionExecutor(
        ExplodingExecutor(),
        policy=policy,
        idempotency_ledger=InMemoryActionExecutionLedger(),
    ).execute(request)
    get_failure = PolicyGuardedActionExecutor(
        SuccessfulExecutor(),
        policy=policy,
        idempotency_ledger=ExplodingLedger(fail_get=True),
    ).execute(request)
    record_failure = PolicyGuardedActionExecutor(
        SuccessfulExecutor(),
        policy=policy,
        idempotency_ledger=ExplodingLedger(fail_record=True),
    ).execute(request)

    assert missing_ledger.status is ActionExecutionStatus.FAILED
    assert "idempotency ledger" in missing_ledger.error
    assert wrapped_failure.status is ActionExecutionStatus.FAILED
    assert wrapped_failure.metadata["possible_side_effects"] is True
    assert "executor boom" in wrapped_failure.error
    assert get_failure.status is ActionExecutionStatus.FAILED
    assert "ledger lookup" in get_failure.error
    assert record_failure.status is ActionExecutionStatus.FAILED
    assert record_failure.metadata["side_effect_status"] == "unknown_after_success"
    assert record_failure.metadata["possible_side_effects"] is True


def test_timeout_action_executor_returns_timed_out_result():
    class SlowExecutor:
        def execute(self, request, context=None):
            time.sleep(0.20)
            return ActionResult(
                action=request.action,
                status=ActionExecutionStatus.SUCCEEDED,
                output={"ok": True},
                metadata={"executor": type(self).__name__, "side_effects": False},
                request_id=request.request_id,
            )

        def execute_many(self, requests, context=None):
            return tuple(self.execute(request, context=context) for request in requests)

    executor = TimeoutActionExecutor(SlowExecutor(), default_timeout_seconds=0.01)
    request = ActionRequest(
        action=ControlAction.RETRIEVE,
        reason="slow evidence fetch",
        request_id="req-timeout",
    )

    try:
        result = executor.execute(request, context={"request_id": "req-timeout"})
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

    assert result.status is ActionExecutionStatus.TIMED_OUT
    assert result.request_id == "req-timeout"
    assert result.metadata["timeout_enforced"] is True
    assert result.metadata["wrapped_executor"] == "SlowExecutor"
    assert result.metadata["side_effects"] is None
    assert result.metadata["side_effect_status"] == "unknown_after_timeout"
    assert result.metadata["possible_side_effects"] is True
    assert "timed out" in result.error


def test_timeout_action_executor_converts_unbounded_executor_exception_to_failed_result():
    class ExplodingExecutor:
        def execute(self, request, context=None):
            raise RuntimeError("boom")

    executor = TimeoutActionExecutor(ExplodingExecutor())
    request = ActionRequest(action=ControlAction.RETRIEVE, reason="explode", request_id="req-timeout-explode")

    try:
        result = executor.execute(request)
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

    assert result.status is ActionExecutionStatus.FAILED
    assert result.request_id == "req-timeout-explode"
    assert result.metadata["wrapped_executor"] == "ExplodingExecutor"
    assert result.metadata["side_effect_status"] == "unknown_after_failure"
    assert result.metadata["possible_side_effects"] is True
    assert "boom" in result.error


def test_timeout_action_executor_preserves_success_metadata():
    class FastExecutor:
        def execute(self, request, context=None):
            return ActionResult(
                action=request.action,
                status=ActionExecutionStatus.SUCCEEDED,
                output={"ok": True},
                metadata={"executor": type(self).__name__, "side_effects": False},
                request_id=request.request_id,
            )

        def execute_many(self, requests, context=None):
            return tuple(self.execute(request, context=context) for request in requests)

    executor = TimeoutActionExecutor(FastExecutor(), default_timeout_seconds=1.0)
    request = ActionRequest(action=ControlAction.RETRIEVE, reason="fast evidence fetch", request_id="req-fast")

    try:
        result = executor.execute(request)
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

    assert result.status is ActionExecutionStatus.SUCCEEDED
    assert result.output == {"ok": True}
    assert result.metadata["executor"] == "FastExecutor"
    assert result.metadata["timeout_wrapper"] == "TimeoutActionExecutor"
    assert result.metadata["timeout_enforced"] is True
    assert result.metadata["timeout_seconds"] == pytest.approx(1.0)


def test_timeout_action_executor_rejects_non_finite_request_timeout():
    class RecordingExecutor:
        def __init__(self):
            self.calls = 0

        def execute(self, request, context=None):
            self.calls += 1
            return ActionResult(
                action=request.action,
                status=ActionExecutionStatus.SUCCEEDED,
                output={"ok": True},
                metadata={"executor": type(self).__name__, "side_effects": False},
                request_id=request.request_id,
            )

        def execute_many(self, requests, context=None):
            return tuple(self.execute(request, context=context) for request in requests)

    wrapped = RecordingExecutor()
    executor = TimeoutActionExecutor(wrapped)
    request = ActionRequest(
        action=ControlAction.RETRIEVE,
        reason="invalid timeout",
        metadata={"timeout_seconds": math.inf},
        request_id="req-invalid-timeout",
    )

    try:
        result = executor.execute(request)
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

    assert result.status is ActionExecutionStatus.FAILED
    assert result.request_id == "req-invalid-timeout"
    assert "positive finite" in result.error
    assert result.metadata["timeout_enforced"] is False
    assert wrapped.calls == 0


def test_timeout_action_executor_rejects_bool_max_workers():
    class FastExecutor:
        def execute(self, request, context=None):
            return ActionResult(action=request.action, status=ActionExecutionStatus.SUCCEEDED)

    with pytest.raises(ValueError, match="max_workers"):
        TimeoutActionExecutor(FastExecutor(), max_workers=True)  # type: ignore[arg-type]


def test_policy_guard_preserves_timeout_enforcement_metadata():
    class FastExecutor:
        def execute(self, request, context=None):
            return ActionResult(
                action=request.action,
                status=ActionExecutionStatus.SUCCEEDED,
                output={"ok": True},
                metadata={"executor": type(self).__name__, "side_effects": False},
                request_id=request.request_id,
            )

        def execute_many(self, requests, context=None):
            return tuple(self.execute(request, context=context) for request in requests)

    timeout_executor = TimeoutActionExecutor(FastExecutor(), default_timeout_seconds=1.0)
    executor = PolicyGuardedActionExecutor(
        timeout_executor,
        policy=ActionExecutionPolicy(max_timeout_seconds=2.0),
    )
    request = ActionRequest(action=ControlAction.RETRIEVE, reason="fast evidence fetch", request_id="req-fast")

    try:
        result = executor.execute(request)
    finally:
        timeout_executor.shutdown(wait=False, cancel_futures=True)

    assert result.status is ActionExecutionStatus.SUCCEEDED
    assert result.metadata["policy_guard"] == "PolicyGuardedActionExecutor"
    assert result.metadata["timeout_wrapper"] == "TimeoutActionExecutor"
    assert result.metadata["timeout_enforced"] is True
    assert result.metadata["timeout_seconds"] == pytest.approx(1.0)


def test_risk_controller_uses_configurable_control_policy():
    artifact = CalibrationArtifact(
        model_id="tiny",
        target_layer=-1,
        scores=(CalibrationScore("maha", threshold=3.0),),
        eigentruth_version="0.1.0",
    )
    controller = RiskController(
        artifact,
        policy_config=ControlPolicyConfig(
            refuted_action=ControlAction.REWRITE,
            unsupported_action=ControlAction.CLARIFY,
            compound_verification_escalates=False,
        ),
    )

    refuted = controller.decide(
        {"maha": 1.0},
        verification_results=(VerificationResult(VerificationStatus.REFUTED, confidence=0.9),),
    )
    unsupported = controller.decide(
        {"maha": 4.0},
        verification_results=(VerificationResult(VerificationStatus.INSUFFICIENT_EVIDENCE, confidence=0.4),),
    )

    assert refuted.action is ControlAction.REWRITE
    assert refuted.risk_level is RiskLevel.HIGH
    assert unsupported.action is ControlAction.CLARIFY
    assert unsupported.risk_level is RiskLevel.MEDIUM


def test_control_policy_config_from_dict_parses_boolean_strings():
    disabled = ControlPolicyConfig.from_dict({"compound_verification_escalates": "false"})
    enabled = ControlPolicyConfig.from_dict({"compound_verification_escalates": "on"})

    assert disabled.compound_verification_escalates is False
    assert enabled.compound_verification_escalates is True
    with pytest.raises(ValueError, match="compound_verification_escalates"):
        ControlPolicyConfig.from_dict({"compound_verification_escalates": "maybe"})


def test_risk_controller_routes_non_finite_diagnostics_to_unknown():
    artifact = CalibrationArtifact(
        model_id="tiny",
        target_layer=-1,
        scores=(CalibrationScore("maha", threshold=3.0),),
        eigentruth_version="0.1.0",
    )
    controller = RiskController(artifact)

    nan_decision = controller.decide({"maha": math.nan})
    inf_decision = controller.decide({"maha": math.inf})

    assert nan_decision.action is ControlAction.CLARIFY
    assert nan_decision.risk_level is RiskLevel.UNKNOWN
    assert nan_decision.diagnostics["invalid_scores"] == ("maha",)
    assert nan_decision.diagnostics["invalid_values"]["maha"] == "nan"
    assert inf_decision.action is ControlAction.CLARIFY
    assert inf_decision.risk_level is RiskLevel.UNKNOWN
    assert inf_decision.diagnostics["invalid_values"]["maha"] == "inf"

    unsupported = controller.decide(
        {"maha": math.nan},
        verification_results=(VerificationResult(VerificationStatus.INSUFFICIENT_EVIDENCE, confidence=0.4),),
    )
    assert unsupported.action is ControlAction.CLARIFY
    assert unsupported.risk_level is RiskLevel.UNKNOWN
    assert unsupported.diagnostics["verification"]["counts"]["insufficient_evidence"] == 1


def test_risk_controller_routes_missing_calibrated_scores_to_unknown():
    artifact = CalibrationArtifact(
        model_id="tiny",
        target_layer=-1,
        scores=(
            CalibrationScore("maha", threshold=3.0),
            CalibrationScore("maha_adaptive", threshold=4.0),
        ),
        eigentruth_version="0.1.0",
    )
    controller = RiskController(artifact)

    decision = controller.decide({"maha": 1.0})

    assert decision.action is ControlAction.CLARIFY
    assert decision.risk_level is RiskLevel.UNKNOWN
    assert decision.diagnostics["missing_scores"] == ("maha_adaptive",)
    assert "missing calibrated diagnostic score" in decision.reason


def test_risk_controller_routes_bool_diagnostics_to_unknown():
    artifact = CalibrationArtifact(
        model_id="tiny",
        target_layer=-1,
        scores=(CalibrationScore("maha", threshold=3.0),),
        eigentruth_version="0.1.0",
    )
    controller = RiskController(artifact)

    decision = controller.decide({"maha": True})

    assert decision.action is ControlAction.CLARIFY
    assert decision.risk_level is RiskLevel.UNKNOWN
    assert decision.diagnostics["invalid_scores"] == ("maha",)
    assert decision.diagnostics["invalid_values"]["maha"] is True


def test_claim_extraction_adds_rule_based_metadata():
    claim = extract_claims("As of 2026, revenue is not 10 dollars [1].")[0]

    features = claim.metadata["features"]

    assert features["has_number"] is True
    assert features["has_citation"] is True
    assert features["has_negation"] is True
    assert features["is_time_sensitive"] is True
    assert features["has_calculation"] is False


def test_claim_extraction_adds_structured_calculation_metadata():
    symbolic, word_operator = extract_claims("3 * 4 = 12. 6 divided by 3 is 3.")

    assert symbolic.metadata["features"]["has_calculation"] is True
    assert symbolic.metadata["calculation"]["expression"] == "3 * 4"
    assert symbolic.metadata["calculation"]["expected"] == 12.0
    assert symbolic.metadata["calculation"]["parser"] == "symbolic"
    assert word_operator.metadata["features"]["has_calculation"] is True
    assert word_operator.metadata["calculation"]["expression"] == "6 / 3"
    assert word_operator.metadata["calculation"]["expected"] == 3.0
    assert word_operator.metadata["calculation"]["parser"] == "word_operator"
    assert extract_calculation("expression: 10 / 2; expected: 5") == {
        "expression": "10 / 2",
        "expected": 5.0,
        "source": "claim_text",
        "parser": "labeled",
    }


def test_groundedness_verifier_uses_claim_metadata_for_failure_reason():
    verifier = GroundednessVerifier(evidence=("AlphaCorp has offices in Europe.",), min_overlap=0.95)
    claim = extract_claims("As of 2026, AlphaCorp has 10 offices.")[0]
    explicit_false = Claim(
        claim.text,
        metadata={"features": {"is_time_sensitive": "false"}},
    )

    result = verifier.verify(claim)
    false_result = verifier.verify(explicit_false)

    assert result.status is VerificationStatus.INSUFFICIENT_EVIDENCE
    assert result.metadata["claim_features"]["is_time_sensitive"] is True
    assert "time-sensitive" in result.explanation
    assert false_result.status is VerificationStatus.INSUFFICIENT_EVIDENCE
    assert false_result.metadata["claim_features"]["is_time_sensitive"] is False
    assert "time-sensitive" not in false_result.explanation


def test_groundedness_evidence_quality_policy_downgrades_stale_time_sensitive_support():
    verifier = GroundednessVerifier(
        evidence=(
            {
                "text": "As of 2026, AlphaCorp has 10 offices.",
                "source": "official.gov/company-registry",
                "timestamp": "2025-01-01",
            },
        ),
        min_overlap=0.7,
        evidence_quality_policy=EvidenceQualityPolicy(
            max_age_days=30,
            reference_time="2026-06-25",
            require_source=True,
            trusted_sources=("official.gov",),
            require_trusted_source=True,
        ),
    )
    claim = extract_claims("As of 2026, AlphaCorp has 10 offices.")[0]

    result = verifier.verify(claim)

    assert result.status is VerificationStatus.INSUFFICIENT_EVIDENCE
    assert result.metadata["decision_rule"] == "evidence_quality_failed"
    assert result.metadata["evidence_quality"]["passed"] is False
    assert "stale_evidence" in result.metadata["evidence_quality"]["reasons"]


def test_groundedness_evidence_quality_policy_allows_fresh_trusted_evidence():
    verifier = GroundednessVerifier(
        evidence=(
            {
                "text": "As of 2026, AlphaCorp has 10 offices.",
                "source": "official.gov/company-registry",
                "metadata": {"published_at": "2026-06-20"},
            },
        ),
        min_overlap=0.7,
        evidence_quality_policy={
            "max_age_days": 30,
            "reference_time": "2026-06-25",
            "require_source": "true",
            "trusted_sources": "official.gov",
            "require_trusted_source": "true",
        },
    )
    claim = extract_claims("As of 2026, AlphaCorp has 10 offices.")[0]

    result = verifier.verify(claim)

    assert result.status is VerificationStatus.SUPPORTED
    assert result.metadata["decision_rule"] == "exact_containment"
    assert result.metadata["evidence_quality"]["passed"] is True
    assert result.metadata["evidence_quality"]["age_days"] == pytest.approx(5.0)


def test_groundedness_evidence_quality_policy_defaults_to_time_sensitive_claims_only():
    verifier = GroundednessVerifier(
        evidence=("Paris is the capital of France.",),
        min_overlap=0.7,
        evidence_quality_policy={
            "max_age_days": 30,
            "reference_time": "2026-06-25",
            "require_source": "true",
            "time_sensitive_only": "true",
        },
    )
    claim = extract_claims("Paris is the capital of France.")[0]

    result = verifier.verify(claim)

    assert result.status is VerificationStatus.SUPPORTED
    assert "evidence_quality" not in result.metadata


def test_evidence_quality_policy_from_dict_strict_bool_parser():
    policy = EvidenceQualityPolicy.from_dict({
        "require_source": "false",
        "require_trusted_source": "0",
        "time_sensitive_only": "off",
    })

    assert policy.require_source is False
    assert policy.require_trusted_source is False
    assert policy.time_sensitive_only is False
    with pytest.raises(ValueError, match="require_source"):
        EvidenceQualityPolicy.from_dict({"require_source": "maybe"})
    with pytest.raises(ValueError, match="max_age_days"):
        EvidenceQualityPolicy.from_dict({"max_age_days": True})
    with pytest.raises(ValueError, match="max_age_days"):
        EvidenceQualityPolicy(max_age_days=1.5)
