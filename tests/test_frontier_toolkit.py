"""Tests for the frontier-toolkit MVP modules."""

import math
import sqlite3

import pytest
import torch

from eigentruth.adapters import (
    CalculatorVerifier,
    InMemoryRetriever,
    InMemoryWorldModelAdapter,
    QuestionAnswerFact,
    QuestionAnswerVerifier,
    RetrievalActionExecutor,
    SQLiteStateQuery,
    SQLiteStateSource,
    StateCheck,
    StructuredStateVerifier,
)
from eigentruth.calibration import CalibrationArtifact, CalibrationScore
from eigentruth.control import (
    ActionExecutionStatus,
    ActionExecutorRegistry,
    ActionRequest,
    ControlAction,
    ControlPolicyConfig,
    DefaultCorrectionPolicy,
    DryRunActionExecutor,
    RiskController,
    RiskDecision,
    RiskLevel,
)
from eigentruth.core import TruthSubspace
from eigentruth.verify import (
    Claim,
    CompositeVerifier,
    EvidenceDocument,
    GroundednessVerifier,
    InMemoryVerifier,
    RoutedVerifier,
    VerificationResult,
    VerificationStatus,
    VerifierRoute,
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
    structured = verifier.verify(
        Claim("The computed total is 12.", metadata={"calculation": {"expression": "3 * 4", "expected": 12}})
    )

    assert supported.status is VerificationStatus.SUPPORTED
    assert supported.metadata["decision_rule"] == "calculation_match"
    assert refuted.status is VerificationStatus.REFUTED
    assert refuted.metadata["decision_rule"] == "calculation_mismatch"
    assert refuted.metadata["actual"] == pytest.approx(4.0)
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
    assert context_route.status is VerificationStatus.REFUTED
    assert context_route.metadata["selected_route"] == "calculator"
    assert fallback_route.status is VerificationStatus.SUPPORTED
    assert fallback_route.metadata["selected_route"] == "fallback"


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
    assert result.metadata["skipped_routes"][0]["status"] == "insufficient_evidence"


def test_in_memory_world_model_adapter_verifies_and_predicts_state():
    verifier = InMemoryVerifier({normalize_claim_text("Inventory is 10"): VerificationStatus.SUPPORTED})
    adapter = InMemoryWorldModelAdapter(verifier=verifier)
    claim = extract_claims("Inventory is 10.")[0]

    result = adapter.verify(claim)
    prediction = adapter.predict({"inventory": 10}, {"set": {"inventory": 8}})

    assert result.status is VerificationStatus.SUPPORTED
    assert prediction.state["inventory"] == 8
    assert "Inventory" in adapter.explain(claim)


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


def test_groundedness_verifier_uses_claim_metadata_for_failure_reason():
    verifier = GroundednessVerifier(evidence=("AlphaCorp has offices in Europe.",), min_overlap=0.95)
    claim = extract_claims("As of 2026, AlphaCorp has 10 offices.")[0]

    result = verifier.verify(claim)

    assert result.status is VerificationStatus.INSUFFICIENT_EVIDENCE
    assert result.metadata["claim_features"]["is_time_sensitive"] is True
    assert "time-sensitive" in result.explanation
