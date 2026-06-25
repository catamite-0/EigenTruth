"""Tests for repository example scripts."""

import importlib
import json
import sqlite3
from types import SimpleNamespace


def test_calibrated_control_demo_defaults_to_best_repository_artifact_when_available():
    demo = importlib.import_module("examples.calibrated_control_demo")

    artifact = demo.default_artifact()
    diagnostics = demo.default_diagnostics_for_artifact(artifact)

    if demo.DEFAULT_SMOLLM2_ARTIFACT_PATH.exists():
        assert artifact.model_id == "HuggingFaceTB/SmolLM2-135M-Instruct"
        assert artifact.target_layer == -16
        assert artifact.score_names() == ("truth_proj",)
        assert diagnostics["truth_proj"] > artifact.get_score("truth_proj").threshold
    elif demo.DEFAULT_QWEN_ARTIFACT_PATH.exists():
        assert artifact.model_id == "Qwen/Qwen2.5-0.5B-Instruct"
        assert artifact.target_layer == -10
    else:
        assert artifact.model_id == "demo-model"
        assert set(diagnostics) == {"maha_last", "subspace_resid"}


def test_calibrated_control_demo_default_trace_uses_artifact_diagnostics():
    demo = importlib.import_module("examples.calibrated_control_demo")

    payload = demo.run(
        SimpleNamespace(
            artifact=None,
            diagnostics=None,
            text=demo.DEFAULT_TEXT,
            facts=None,
            evidence=None,
            refutations=None,
            retrieval_evidence=None,
            enable_calculator=False,
            calculator_context=None,
            runtime_profile=None,
            staged_verification=None,
            request_id="test-demo",
            output=None,
            registry=None,
        )
    )

    assert payload["metadata"]["artifact_model_id"] == demo.default_artifact().model_id
    assert payload["metadata"]["artifact_source"] == demo.artifact_source(None)
    if demo.default_promotion_contract_path() is not None:
        assert "v1_5" in payload["metadata"]["promotion_contract_source"]
        assert payload["metadata"]["promotion_contract_model_id"] == "HuggingFaceTB/SmolLM2-135M-Instruct"
        assert payload["metadata"]["promotion_contract_budget_enabled"] is False
        assert payload["metadata"]["promotion_contract_metadata"]["recommended_performance_baseline_record"] == (
            "performance_baseline:smollm2-l20-performance-baseline:0.9"
        )
        assert payload["metadata"]["promotion_contract_metadata"]["recommended_selector_replay_candidate"] == "default"
        assert payload["metadata"]["promotion_contract_metadata"]["selector_replay_status"] == "promote"
        assert payload["metadata"]["promotion_contract_metadata"]["product_runtime_drift_status"] == "promote"
        assert payload["metadata"]["promotion_contract_metadata"]["product_runtime_drift_blocked_metric_count"] == 0
        assert payload["metadata"]["promotion_contract_metadata"]["adapter_family_required_routes"] == [
            "structured_state",
            "state_transition",
            "retrieval_groundedness",
            "retrieval_structured_qa",
        ]
        assert payload["metadata"]["promotion_contract_metadata"]["required_route_baseline_routes"] == [
            "retrieval_structured_qa"
        ]
    for score_name in demo.default_artifact().score_names():
        assert score_name in payload["diagnostics"]
    assert payload["risk_decision"]["action"] == "abstain"
    assert payload["runtime_trace"]["summary"]["phase_counts"]["action_execution"] == 1


def test_calibrated_control_demo_can_emit_bounded_trace():
    demo = importlib.import_module("examples.calibrated_control_demo")

    payload = demo.run(
        SimpleNamespace(
            artifact=None,
            diagnostics=None,
            text=demo.DEFAULT_TEXT,
            facts=None,
            evidence=None,
            refutations=None,
            retrieval_evidence=None,
            enable_calculator=False,
            calculator_context=None,
            runtime_profile=None,
            staged_verification=None,
            request_id="test-bounded-demo",
            output=None,
            registry=None,
            bounded_trace=True,
            bounded_trace_max_claims=1,
            bounded_trace_max_verification_results=1,
            bounded_trace_max_actions=1,
            bounded_trace_max_action_results=1,
            bounded_trace_max_events=2,
            bounded_trace_max_nested_items=4,
            bounded_trace_include_runtime_trace=False,
        )
    )

    assert payload["trace_format"] == "bounded_product_trace"
    assert payload["request_id"] == "test-bounded-demo"
    assert payload["risk_decision"]["action"] == "abstain"
    assert payload["runtime_trace"] is None
    assert payload["truncation"]["runtime_trace_included"] is False
    assert payload["summaries"]["runtime"]["measured_phases"] > 0
    assert payload["truncation"]["claims"]["included"] <= 1
    assert payload["truncation"]["verification_results"]["included"] <= 1
    assert payload["metadata"]["artifact_source"] == demo.artifact_source(None)
    if demo.default_promotion_contract_path() is not None:
        assert "v1_5" in payload["metadata"]["promotion_contract_source"]


def test_calibrated_control_demo_can_route_calculator_refutations():
    demo = importlib.import_module("examples.calibrated_control_demo")

    payload = demo.run(
        SimpleNamespace(
            artifact=None,
            diagnostics='{"truth_proj": 0.0}',
            text="2 + 2 = 5.",
            facts=None,
            evidence=None,
            refutations=None,
            retrieval_evidence=None,
            enable_calculator=True,
            calculator_context=None,
            runtime_profile=None,
            staged_verification=None,
            request_id="test-calculator-demo",
            output=None,
            registry=None,
        )
    )

    result = payload["verification_results"][0]

    assert payload["metadata"]["calculator_enabled"] is True
    assert payload["metadata"]["verifier_type"] == "RoutedVerifier"
    assert result["status"] == "refuted"
    assert result["metadata"]["selected_route"] == "calculator"
    assert result["metadata"]["selected_verifier"] == "CalculatorVerifier"
    assert payload["risk_decision"]["action"] == "abstain"


def test_calibrated_control_demo_can_write_compact_trace_json(tmp_path):
    demo = importlib.import_module("examples.calibrated_control_demo")
    output_path = tmp_path / "trace.json"

    payload = demo.run(
        SimpleNamespace(
            artifact=None,
            diagnostics='{"truth_proj": 0.0}',
            text="Paris is the capital of France.",
            facts='{"Paris is the capital of France": "supported"}',
            evidence=None,
            refutations=None,
            retrieval_evidence=None,
            enable_calculator=False,
            calculator_context=None,
            runtime_profile=None,
            staged_verification=None,
            runtime_trace=True,
            promotion_contract=None,
            request_id="compact-demo",
            output=str(output_path),
            registry=None,
            compact_json=True,
        )
    )
    written = output_path.read_text(encoding="utf-8")

    assert json.loads(written)["request_id"] == payload["request_id"]
    assert "\n  " not in written


def test_calibrated_control_demo_latency_profile_skips_low_risk_non_sensitive_verification():
    demo = importlib.import_module("examples.calibrated_control_demo")
    artifact = demo.default_artifact()

    payload = demo.run(
        SimpleNamespace(
            artifact=None,
            diagnostics=json.dumps(demo.low_diagnostics_for_artifact(artifact)),
            text="Paris is the capital of France. The moon is made of cheese.",
            facts=None,
            evidence=None,
            refutations=None,
            retrieval_evidence=None,
            enable_calculator=False,
            calculator_context=None,
            runtime_profile="latency",
            staged_verification=None,
            request_id="test-latency-profile-demo",
            output=None,
            registry=None,
        )
    )

    stage_event = next(event for event in payload["events"] if event["event_type"] == "verification_stage_decision")

    assert payload["metadata"]["runtime_profile"] == "latency"
    assert payload["metadata"]["staged_verification_enabled"] is True
    assert payload["metadata"]["staged_verification"]["verify_triggered_claims_only"] is True
    assert payload["metadata"]["max_verifier_route_attempts"] == 1
    assert payload["verification_results"] == []
    assert "initial_verification" not in {
        phase["name"] for phase in payload["runtime_trace"]["phases"]
    }
    assert stage_event["payload"]["run_verifier"] is False
    assert stage_event["payload"]["reason"] == "diagnostics and claim metadata did not require verification"
    assert payload["metadata"]["verification_stage_summary"]["skipped"] is True
    assert payload["metadata"]["verification_stage_summary"]["saved_claim_count"] == 2
    assert payload["metadata"]["staged_verification"]["fail_closed_on_skip"] is True
    assert payload["risk_decision"]["action"] == "clarify"
    assert payload["risk_decision"]["risk_level"] == "unknown"


def test_calibrated_control_demo_can_disable_staged_verification_from_string_default():
    demo = importlib.import_module("examples.calibrated_control_demo")

    policy = demo.stage_policy_from_runtime_profile(
        None,
        staged_verification=None,
        control_defaults={"staged_verification": "false"},
    )

    assert policy is None


def test_calibrated_control_demo_can_opt_out_of_fail_closed_staged_skip():
    demo = importlib.import_module("examples.calibrated_control_demo")

    policy = demo.stage_policy_from_runtime_profile(
        None,
        staged_verification=None,
        control_defaults={
            "staged_verification": "true",
            "stage_fail_closed_on_skip": "false",
        },
    )

    assert policy is not None
    assert policy.fail_closed_on_skip is False


def test_calibrated_control_demo_balanced_profile_verifies_diagnostic_risk():
    demo = importlib.import_module("examples.calibrated_control_demo")

    payload = demo.run(
        SimpleNamespace(
            artifact=None,
            diagnostics=None,
            text=demo.DEFAULT_TEXT,
            facts=None,
            evidence=None,
            refutations=None,
            retrieval_evidence=None,
            enable_calculator=False,
            calculator_context=None,
            runtime_profile="balanced",
            staged_verification=None,
            request_id="test-balanced-profile-demo",
            output=None,
            registry=None,
        )
    )

    stage_event = next(event for event in payload["events"] if event["event_type"] == "verification_stage_decision")

    assert payload["metadata"]["runtime_profile"] == "balanced"
    assert payload["metadata"]["staged_verification_enabled"] is True
    assert stage_event["payload"]["run_verifier"] is True
    assert payload["verification_results"][1]["status"] == "refuted"
    assert payload["risk_decision"]["action"] == "abstain"


def test_calibrated_control_demo_auto_profile_selects_latency_or_audit():
    demo = importlib.import_module("examples.calibrated_control_demo")
    artifact = demo.default_artifact()
    low_diagnostics = json.dumps(demo.low_diagnostics_for_artifact(artifact))

    latency_payload = demo.run(
        SimpleNamespace(
            artifact=None,
            diagnostics=low_diagnostics,
            text="Paris is the capital of France.",
            facts='{"Paris is the capital of France": "supported"}',
            evidence=None,
            refutations=None,
            retrieval_evidence=None,
            enable_calculator=False,
            calculator_context=None,
            runtime_profile="auto",
            staged_verification=None,
            runtime_trace=True,
            request_id="auto-latency",
            output=None,
            registry=None,
        )
    )
    audit_payload = demo.run(
        SimpleNamespace(
            artifact=None,
            diagnostics=low_diagnostics,
            text="2 + 2 = 4.",
            facts=None,
            evidence=None,
            refutations=None,
            retrieval_evidence=None,
            enable_calculator=True,
            calculator_context=None,
            runtime_profile="auto",
            staged_verification=None,
            runtime_trace=True,
            request_id="auto-audit",
            output=None,
            registry=None,
        )
    )

    assert latency_payload["metadata"]["runtime_profile"] == "latency"
    assert latency_payload["metadata"]["runtime_profile_selection"]["selected_profile"] == "latency"
    assert latency_payload["metadata"]["verification_stage_summary"]["skipped"] is True
    assert audit_payload["metadata"]["runtime_profile"] == "audit"
    assert audit_payload["metadata"]["runtime_profile_selection"]["selected_profile"] == "audit"
    assert audit_payload["metadata"]["runtime_profile_selection"]["triggered_claim_ids"] == ("c1",)
    assert audit_payload["metadata"]["verification_stage_summary"]["skipped"] is False


def test_calibrated_control_demo_auto_profile_uses_selector_policy(tmp_path):
    demo = importlib.import_module("examples.calibrated_control_demo")
    artifact = demo.default_artifact()
    low_diagnostics = json.dumps(demo.low_diagnostics_for_artifact(artifact))
    policy_path = tmp_path / "selector-policy.json"
    policy_path.write_text(
        json.dumps({
            "sensitive_claim_feature_flags": ["has_citation"],
            "sensitive_claim_metadata_keys": ["requires_review"],
        }),
        encoding="utf-8",
    )

    payload = demo.run(
        SimpleNamespace(
            artifact=None,
            diagnostics=low_diagnostics,
            text="2 + 2 = 4.",
            facts='{"2 + 2 = 4": "supported"}',
            evidence=None,
            refutations=None,
            retrieval_evidence=None,
            enable_calculator=False,
            calculator_context=None,
            runtime_profile="auto",
            runtime_profile_selector_policy=str(policy_path),
            staged_verification=None,
            runtime_trace=True,
            request_id="auto-selector-policy",
            output=None,
            registry=None,
        )
    )

    assert payload["metadata"]["runtime_profile"] == "latency"
    assert payload["metadata"]["runtime_profile_selection"]["selected_profile"] == "latency"
    assert payload["metadata"]["runtime_profile_selector_policy"]["sensitive_claim_feature_flags"] == (
        "has_citation",
    )


def test_calibrated_control_demo_pre_generation_profile_records_and_applies():
    demo = importlib.import_module("examples.calibrated_control_demo")
    artifact = demo.default_artifact()

    payload = demo.run(
        SimpleNamespace(
            artifact=None,
            diagnostics=json.dumps(demo.low_diagnostics_for_artifact(artifact)),
            text="The latest BTC price today is 100 dollars.",
            facts=None,
            evidence=None,
            refutations=None,
            retrieval_evidence=None,
            enable_calculator=False,
            calculator_context=None,
            runtime_profile=None,
            runtime_profile_selector_policy=None,
            pre_generation_profile="auto",
            pre_generation_risk_policy=None,
            pre_generation_metadata='{"requires_current_facts": "yes"}',
            staged_verification=None,
            runtime_trace=True,
            request_id="pre-generation-auto",
            output=None,
            registry=None,
        )
    )

    assessment = payload["metadata"]["pre_generation_risk_assessment"]

    assert payload["metadata"]["runtime_profile"] == "audit"
    assert payload["metadata"]["runtime_profile_source"] == "pre_generation"
    assert payload["metadata"]["pre_generation_profile_requested"] == "auto"
    assert assessment["selected_profile"] == "audit"
    assert assessment["risk_level"] == "high"
    assert "is_time_sensitive" in assessment["triggered_features"]
    assert assessment["triggered_metadata"] == ("requires_current_facts",)
    assert payload["metadata"]["staged_verification_enabled"] is False


def test_calibrated_control_demo_uses_promotion_contract_release_efficiency_profile(tmp_path):
    demo = importlib.import_module("examples.calibrated_control_demo")
    from eigentruth.control import ProductPromotionContract

    artifact = demo.default_artifact()
    contract_path = tmp_path / "promotion-contract.json"
    ProductPromotionContract(
        model_id="demo-model",
        runtime={"layer": -1},
        verifier_route={"route": "fallback"},
        release_efficiency={
            "status": "promote",
            "recommended_profile": "latency",
            "recommended_efficiency_score": 2.0,
        },
        source_status="promote",
    ).save_json(contract_path)

    payload = demo.run(
        SimpleNamespace(
            artifact=None,
            diagnostics=json.dumps(demo.low_diagnostics_for_artifact(artifact)),
            text="Paris is the capital of France.",
            facts='{"Paris is the capital of France": "supported"}',
            evidence=None,
            refutations=None,
            retrieval_evidence=None,
            enable_calculator=False,
            calculator_context=None,
            runtime_profile=None,
            pre_generation_profile="off",
            staged_verification=None,
            runtime_trace=True,
            promotion_contract=str(contract_path),
            request_id="contract-release-efficiency-profile",
            output=None,
            registry=None,
        )
    )

    assert payload["metadata"]["runtime_profile"] == "latency"
    assert payload["metadata"]["runtime_profile_source"] == "promotion_contract_release_efficiency"
    assert payload["metadata"]["promotion_contract_release_efficiency"] == {
        "status": "promote",
        "recommended_profile": "latency",
        "recommended_efficiency_score": 2.0,
    }
    assert payload["metadata"]["staged_verification_enabled"] is True
    assert payload["metadata"]["verification_stage_summary"]["skipped"] is True
    assert payload["metadata"]["staged_verification"]["fail_closed_on_skip"] is True
    assert payload["risk_decision"]["action"] == "clarify"


def test_calibrated_control_demo_can_record_runtime_budget_result():
    demo = importlib.import_module("examples.calibrated_control_demo")

    payload = demo.run(
        SimpleNamespace(
            artifact=None,
            diagnostics='{"truth_proj": 0.0}',
            text="Paris is the capital of France.",
            facts='{"Paris is the capital of France": "supported"}',
            evidence=None,
            refutations=None,
            retrieval_evidence=None,
            enable_calculator=False,
            calculator_context=None,
            runtime_profile=None,
            staged_verification=None,
            runtime_trace=False,
            max_runtime_total_seconds=1.0,
            max_runtime_phase_seconds=None,
            request_id="test-runtime-budget-demo",
            output=None,
            registry=None,
        )
    )

    runtime_budget = payload["metadata"]["runtime_budget"]
    assert payload["runtime_trace"] is None
    assert runtime_budget["enabled"] is True
    assert runtime_budget["passed"] is False
    assert runtime_budget["failures"][0]["metric"] == "runtime_trace"


def test_calibrated_control_demo_records_cache_summary_and_budget():
    demo = importlib.import_module("examples.calibrated_control_demo")

    payload = demo.run(
        SimpleNamespace(
            artifact=None,
            diagnostics='{"truth_proj": 0.0}',
            text="Paris is the capital of France.",
            facts='{"Paris is the capital of France": "supported"}',
            evidence=None,
            refutations=None,
            retrieval_evidence=None,
            enable_calculator=False,
            calculator_context=None,
            runtime_profile=None,
            staged_verification=None,
            runtime_trace=True,
            cache_verifier=True,
            cache_retriever=False,
            max_runtime_total_seconds=None,
            max_runtime_phase_seconds=None,
            min_cache_hit_rate=0.5,
            min_named_cache_hit_rate=None,
            request_id="test-cache-budget-demo",
            output=None,
            registry=None,
        )
    )

    cache_summary = payload["metadata"]["cache_summary"]
    runtime_budget = payload["metadata"]["runtime_budget"]

    assert payload["metadata"]["cache"]["verifier"]["misses"] == 1
    assert cache_summary["aggregate"]["requests"] == 1
    assert cache_summary["aggregate"]["hit_rate"] == 0.0
    assert runtime_budget["passed"] is False
    assert runtime_budget["failures"][0]["metric"] == "cache_hit_rate"


def test_calibrated_control_demo_records_route_cost_summary_and_budget():
    demo = importlib.import_module("examples.calibrated_control_demo")

    payload = demo.run(
        SimpleNamespace(
            artifact=None,
            diagnostics='{"truth_proj": 0.0}',
            text="Paris is the capital of France.",
            facts='{"Paris is the capital of France": "supported"}',
            evidence=None,
            refutations=None,
            retrieval_evidence=None,
            enable_calculator=True,
            calculator_context=None,
            runtime_profile=None,
            staged_verification=None,
            runtime_trace=True,
            max_runtime_total_seconds=None,
            max_runtime_phase_seconds=None,
            max_mean_route_duration_seconds=None,
            max_p95_route_duration_seconds=None,
            max_p99_route_duration_seconds=None,
            max_mean_attempted_route_count=0.5,
            max_retrieval_use_rate=None,
            min_cache_hit_rate=None,
            min_named_cache_hit_rate=None,
            request_id="test-route-cost-budget-demo",
            output=None,
            registry=None,
        )
    )

    route_cost_summary = payload["metadata"]["route_cost_summary"]
    runtime_budget = payload["metadata"]["runtime_budget"]

    assert route_cost_summary["routed_total"] == 1
    assert route_cost_summary["mean_attempted_route_count"] == 1.0
    assert runtime_budget["passed"] is False
    assert runtime_budget["failures"][0]["metric"] == "mean_attempted_route_count"


def test_calibrated_control_demo_can_use_promotion_contract_budget(tmp_path):
    demo = importlib.import_module("examples.calibrated_control_demo")
    from eigentruth.control import ProductPromotionContract, ProductRuntimeBudgetPolicy

    contract_path = tmp_path / "promotion-contract.json"
    ProductPromotionContract(
        model_id="demo-model",
        runtime={"layer": -1},
        verifier_route={"route": "fallback"},
        runtime_budget_policy=ProductRuntimeBudgetPolicy(max_mean_attempted_route_count=0.5),
        source_status="promote",
    ).save_json(contract_path)

    payload = demo.run(
        SimpleNamespace(
            artifact=None,
            diagnostics='{"truth_proj": 0.0}',
            text="Paris is the capital of France.",
            facts='{"Paris is the capital of France": "supported"}',
            evidence=None,
            refutations=None,
            retrieval_evidence=None,
            enable_calculator=True,
            calculator_context=None,
            runtime_profile=None,
            staged_verification=None,
            runtime_trace=True,
            promotion_contract=str(contract_path),
            max_runtime_total_seconds=None,
            max_runtime_phase_seconds=None,
            max_mean_route_duration_seconds=None,
            max_p95_route_duration_seconds=None,
            max_p99_route_duration_seconds=None,
            max_route_duration_seconds=None,
            max_mean_attempted_route_count=None,
            max_retrieval_use_rate=None,
            max_retrieval_hit_count=None,
            min_cache_hit_rate=None,
            min_named_cache_hit_rate=None,
            request_id="test-promotion-contract-demo",
            output=None,
            registry=None,
        )
    )

    runtime_budget = payload["metadata"]["runtime_budget"]

    assert runtime_budget["policy"]["max_mean_attempted_route_count"] == 0.5
    assert runtime_budget["passed"] is False
    assert runtime_budget["failures"][0]["metric"] == "mean_attempted_route_count"


def test_calibrated_control_demo_applies_promotion_contract_control_defaults(tmp_path):
    demo = importlib.import_module("examples.calibrated_control_demo")
    from eigentruth.control import ProductPromotionContract

    contract_path = tmp_path / "promotion-contract.json"
    ProductPromotionContract(
        model_id="demo-model",
        runtime={"layer": -1},
        verifier_route={"route": "fallback"},
        control_defaults={"max_verifier_route_attempts": 1},
        source_status="promote",
    ).save_json(contract_path)

    base_args = dict(
        artifact=None,
        diagnostics='{"truth_proj": 0.0}',
        text="Paris is the capital of France.",
        facts='{"Paris is the capital of France": "supported"}',
        evidence=None,
        refutations=None,
        retrieval_evidence=None,
        enable_calculator=False,
        calculator_context=None,
        runtime_profile="balanced",
        staged_verification=None,
        runtime_trace=True,
        promotion_contract=str(contract_path),
        output=None,
        registry=None,
    )

    contract_default_payload = demo.run(SimpleNamespace(
        **base_args,
        max_verifier_route_attempts=None,
        request_id="contract-control-default",
    ))
    explicit_payload = demo.run(SimpleNamespace(
        **base_args,
        max_verifier_route_attempts=4,
        request_id="explicit-control-default",
    ))

    assert contract_default_payload["metadata"]["runtime_profile"] == "balanced"
    assert contract_default_payload["metadata"]["runtime_profile_control_defaults"][
        "max_verifier_route_attempts"
    ] == 2
    assert contract_default_payload["metadata"]["promotion_contract_control_defaults"] == {
        "max_verifier_route_attempts": 1
    }
    assert contract_default_payload["metadata"]["effective_control_defaults"][
        "max_verifier_route_attempts"
    ] == 1
    assert contract_default_payload["metadata"]["max_verifier_route_attempts"] == 1
    assert contract_default_payload["metadata"]["verifier_type"] == "RoutedVerifier"
    assert explicit_payload["metadata"]["effective_control_defaults"]["max_verifier_route_attempts"] == 1
    assert explicit_payload["metadata"]["max_verifier_route_attempts"] == 4


def test_calibrated_control_demo_applies_promotion_contract_control_policy(tmp_path):
    demo = importlib.import_module("examples.calibrated_control_demo")
    from eigentruth.control import ProductPromotionContract

    contract_path = tmp_path / "promotion-contract.json"
    ProductPromotionContract(
        model_id="demo-model",
        runtime={"layer": -1},
        verifier_route={"route": "fallback"},
        control_policy_config={
            "unsupported_action": "clarify",
            "compound_verification_escalates": False,
        },
        feedback_policy_workflow={
            "report_path": "feedback-policy-workflow.json",
            "promotion_decision": "promote_candidate_policy",
        },
        source_status="promote",
    ).save_json(contract_path)

    payload = demo.run(
        SimpleNamespace(
            artifact=None,
            diagnostics='{"truth_proj": 0.0}',
            text="Unverified revenue increased by 12 percent.",
            facts="{}",
            evidence=None,
            refutations=None,
            retrieval_evidence=None,
            enable_calculator=False,
            calculator_context=None,
            runtime_profile="balanced",
            staged_verification=None,
            runtime_trace=True,
            bounded_trace=True,
            bounded_trace_max_claims=20,
            bounded_trace_max_verification_results=20,
            bounded_trace_max_actions=20,
            bounded_trace_max_action_results=20,
            bounded_trace_max_events=20,
            bounded_trace_max_nested_items=16,
            bounded_trace_include_runtime_trace=False,
            promotion_contract=str(contract_path),
            max_verifier_route_attempts=None,
            request_id="contract-control-policy",
            output=None,
            registry=None,
        )
    )

    assert payload["risk_decision"]["action"] == "clarify"
    assert payload["metadata"]["control_policy_source"] == "promotion_contract_feedback_policy"
    assert payload["metadata"]["effective_control_policy_config"]["unsupported_action"] == "clarify"
    assert payload["metadata"]["promotion_contract_control_policy_config"][
        "unsupported_action"
    ] == "clarify"
    assert payload["metadata"]["promotion_contract_feedback_policy_workflow"][
        "promotion_decision"
    ] == "promote_candidate_policy"


def test_calibrated_control_demo_can_use_default_structured_retrieval_audit_contract_budget():
    demo = importlib.import_module("examples.calibrated_control_demo")
    contract_path = demo.default_promotion_contract_path()
    assert contract_path is not None
    assert "smollm2_product_promotion_contract_v1_5" in str(contract_path)
    assert contract_path.name == "product-promotion-contract.json"

    payload = demo.run(
        SimpleNamespace(
            artifact=None,
            diagnostics='{"truth_proj": 0.0}',
            text="Paris is the capital of France.",
            facts='{"Paris is the capital of France": "supported"}',
            evidence=None,
            refutations=None,
            retrieval_evidence=None,
            enable_calculator=False,
            calculator_context=None,
            runtime_profile=None,
            staged_verification=None,
            runtime_trace=True,
            promotion_contract=str(contract_path),
            promotion_contract_manifest=None,
            promotion_contract_registry="artifacts/local-release-registry.json",
            promotion_contract_registry_key=None,
            verify_promotion_contract_manifest=True,
            cache_verifier=False,
            cache_retriever=False,
            max_runtime_total_seconds=None,
            max_runtime_phase_seconds=None,
            max_runtime_phase_p95_seconds=None,
            max_runtime_phase_p99_seconds=None,
            max_mean_route_duration_seconds=None,
            max_p95_route_duration_seconds=None,
            max_p99_route_duration_seconds=None,
            max_route_duration_seconds=None,
            max_mean_attempted_route_count=None,
            max_retrieval_use_rate=None,
            max_retrieval_hit_count=None,
            min_cache_hit_rate=None,
            min_named_cache_hit_rate=None,
            request_id="test-default-promotion-contract-demo",
            output=None,
            registry=None,
        )
    )

    runtime_budget = payload["metadata"]["runtime_budget"]
    route_summary = payload["metadata"]["route_cost_summary"]

    assert payload["metadata"]["promotion_contract_budget_enabled"] is True
    assert payload["metadata"]["promotion_contract_manifest"].endswith("artifact-manifest.json")
    assert payload["metadata"]["promotion_contract_manifest_verification"]["passed"] is True
    assert payload["metadata"]["promotion_contract_manifest_verification"]["checked"] == 2
    assert payload["metadata"]["promotion_contract_registry"] == "artifacts/local-release-registry.json"
    assert payload["metadata"]["promotion_contract_registry_key"] == (
        "product_promotion_contract:smollm2-product-promotion-contract:1.5"
    )
    assert payload["metadata"]["promotion_contract_metadata"]["recommended_performance_baseline_record"] == (
        "performance_baseline:smollm2-l20-performance-baseline:0.9"
    )
    assert payload["metadata"]["promotion_contract_metadata"]["recommended_selector_replay_candidate"] == "default"
    assert payload["metadata"]["promotion_contract_metadata"]["selector_replay_status"] == "promote"
    assert payload["metadata"]["promotion_contract_metadata"]["product_runtime_drift_status"] == "promote"
    assert payload["metadata"]["promotion_contract_metadata"]["product_runtime_drift_blocked_metric_count"] == 0
    assert payload["metadata"]["promotion_contract_metadata"]["adapter_family_promotion_status"] == "promote"
    assert payload["metadata"]["promotion_contract_metadata"]["adapter_family_required_routes"] == [
        "structured_state",
        "state_transition",
        "retrieval_groundedness",
        "retrieval_structured_qa",
    ]
    assert payload["metadata"]["promotion_contract_metadata"]["required_route_baseline_status"] == "promote"
    assert payload["metadata"]["promotion_contract_metadata"]["required_route_baseline_records"] == [
        "benchmark_manifest:smollm2-l80-retrieval-structured-qa-route:0.5"
    ]
    assert payload["metadata"]["promotion_contract_metadata"]["required_route_baseline_routes"] == [
        "retrieval_structured_qa"
    ]
    assert payload["metadata"]["promotion_contract_metadata"]["required_route_budget_policy"][
        "required_route_max_retrieval_hit_count"
    ] == 450.0
    assert payload["verification_results"][0]["metadata"]["selected_route"] == "structured_qa"
    assert route_summary["mean_attempted_route_count"] == 1.0
    assert runtime_budget["passed"] is True
    assert runtime_budget["policy"]["max_mean_attempted_route_count"] == 1.1
    assert runtime_budget["policy"]["max_retrieval_use_rate"] == 0.0


def test_sqlite_state_control_demo_refutes_database_state_claim(tmp_path):
    demo = importlib.import_module("examples.sqlite_state_control_demo")

    payload = demo.run(
        SimpleNamespace(
            database=str(tmp_path / "orders.db"),
            seed_database=True,
            diagnostics=None,
            request_id="test-sqlite-state-demo",
            output=None,
        )
    )

    statuses = [result["status"] for result in payload["verification_results"]]
    routes = [result["metadata"]["decision_rule"] for result in payload["verification_results"]]

    assert payload["metadata"]["state_source_type"] == "SQLiteStateSource"
    assert payload["metadata"]["business_domain"] == "order_fulfillment"
    assert payload["diagnostics"] == {"truth_proj": 0.0}
    assert statuses == ["supported", "refuted"]
    assert routes == ["state_check_passed", "state_check_failed"]
    assert payload["risk_decision"]["action"] == "abstain"
    assert payload["risk_decision"]["risk_level"] == "high"
    assert payload["actions"][0]["action"] == "abstain"
    assert payload["action_results"][0]["status"] == "dry_run"
    assert payload["action_results"][0]["output"]["would_execute"] == "abstain"


def test_state_transition_control_demo_refutes_predicted_postcondition():
    demo = importlib.import_module("examples.state_transition_control_demo")

    payload = demo.run(
        SimpleNamespace(
            diagnostics=None,
            state=None,
            request_id="test-state-transition-demo",
            output=None,
        )
    )

    statuses = [result["status"] for result in payload["verification_results"]]
    routes = [result["metadata"]["decision_rule"] for result in payload["verification_results"]]

    assert payload["metadata"]["verifier_type"] == "StateTransitionVerifier"
    assert payload["metadata"]["world_model_type"] == "InMemoryWorldModelAdapter"
    assert payload["metadata"]["business_domain"] == "order_fulfillment_transition"
    assert payload["diagnostics"] == {"truth_proj": 0.0}
    assert statuses == ["supported", "refuted"]
    assert routes == ["transition_postcondition_passed", "transition_postcondition_failed"]
    assert payload["risk_decision"]["action"] == "abstain"
    assert payload["risk_decision"]["risk_level"] == "high"


def test_production_tool_loop_demo_maps_tool_output_to_postcondition(tmp_path):
    demo = importlib.import_module("examples.production_tool_loop_demo")

    payload = demo.run(
        SimpleNamespace(
            database=str(tmp_path / "orders.db"),
            seed_database=True,
            diagnostics=None,
            tool_input=None,
            request_id="test-production-tool-loop",
            execution_ledger=None,
            output=None,
        )
    )

    statuses = [result["status"] for result in payload["verification_results"]]
    selected_routes = [
        result["metadata"]["selected_route"]
        for result in payload["verification_results"]
    ]
    route_summary = payload["metadata"]["route_summary"]

    assert payload["metadata"]["business_domain"] == "order_reservation"
    assert payload["metadata"]["tool"] == "reserve_inventory"
    assert payload["diagnostics"] == {"truth_proj": 0.0}
    assert payload["actions"][0]["action"] == "execute_tool"
    assert payload["actions"][0]["payload"]["tool"] == "reserve_inventory"
    assert payload["action_results"][0]["status"] == "succeeded"
    assert payload["action_results"][0]["metadata"]["side_effects"] is True
    assert payload["action_results"][0]["metadata"]["policy_guard"] == "PolicyGuardedActionExecutor"
    assert payload["action_results"][0]["metadata"]["idempotency_key"] == (
        "reserve_inventory:test-production-tool-loop:ord_1"
    )
    assert payload["action_results"][0]["metadata"]["timeout_seconds"] == 5.0
    assert payload["action_results"][0]["metadata"]["timeout_enforced"] is False
    assert payload["action_results"][0]["metadata"]["execution_policy"]["require_idempotency_key"] is True
    assert payload["action_results"][0]["output"]["remaining"] == 7
    assert statuses == ["supported", "supported", "refuted"]
    assert selected_routes == ["database_state", "tool_output_state", "tool_output_state"]
    assert route_summary["counts_by_selected_route"] == {
        "database_state": 1,
        "tool_output_state": 2,
    }
    assert route_summary["counts_by_status"] == {"supported": 2, "refuted": 1}
    assert payload["metadata"]["action_execution_summary"]["side_effects"] is True
    assert payload["verification_results"][1]["metadata"]["actual"] == 7
    assert payload["verification_results"][2]["metadata"]["actual"] is False
    assert payload["risk_decision"]["action"] == "abstain"
    assert payload["risk_decision"]["risk_level"] == "high"


def test_production_tool_loop_demo_records_failed_tool_without_side_effect(tmp_path):
    demo = importlib.import_module("examples.production_tool_loop_demo")

    payload = demo.run(
        SimpleNamespace(
            database=str(tmp_path / "orders.db"),
            seed_database=True,
            diagnostics=None,
            tool_input='{"order_id":"missing"}',
            request_id="test-production-tool-loop-failed-tool",
            execution_ledger=None,
            output=None,
        )
    )

    statuses = [result["status"] for result in payload["verification_results"]]
    tool_event = next(event for event in payload["events"] if event["event_type"] == "local_tool_executed")

    assert payload["action_results"][0]["status"] == "failed"
    assert payload["action_results"][0]["metadata"]["side_effects"] is False
    assert payload["action_results"][0]["metadata"]["policy_guard"] == "PolicyGuardedActionExecutor"
    assert payload["action_results"][0]["metadata"]["idempotency_key"] == (
        "reserve_inventory:test-production-tool-loop-failed-tool:missing"
    )
    assert payload["metadata"]["action_execution_summary"]["side_effects"] is False
    assert tool_event["payload"]["status"] == "failed"
    assert tool_event["payload"]["side_effects"] is False
    assert statuses == ["supported", "insufficient_evidence", "insufficient_evidence"]
    assert payload["verification_results"][1]["metadata"]["decision_rule"] == "tool_output_missing"


def test_production_tool_loop_demo_replays_from_execution_ledger_without_second_mutation(tmp_path):
    demo = importlib.import_module("examples.production_tool_loop_demo")
    database_path = tmp_path / "orders.db"
    ledger_path = tmp_path / "action-ledger.sqlite"

    first = demo.run(
        SimpleNamespace(
            database=str(database_path),
            seed_database=True,
            diagnostics=None,
            tool_input=None,
            request_id="test-production-tool-loop-replay",
            execution_ledger=str(ledger_path),
            execution_ledger_backend="sqlite",
            output=None,
        )
    )
    second = demo.run(
        SimpleNamespace(
            database=str(database_path),
            seed_database=False,
            diagnostics=None,
            tool_input=None,
            request_id="test-production-tool-loop-replay",
            execution_ledger=str(ledger_path),
            execution_ledger_backend="sqlite",
            output=None,
        )
    )
    connection = sqlite3.connect(database_path)
    try:
        available = connection.execute("select available from inventory where sku = ?", ("sku_123",)).fetchone()[0]
    finally:
        connection.close()

    assert first["action_results"][0]["metadata"]["idempotency_replayed"] is False
    assert first["action_results"][0]["metadata"]["side_effects"] is True
    assert first["metadata"]["execution_ledger_backend"] == "sqlite"
    assert second["action_results"][0]["metadata"]["idempotency_replayed"] is True
    assert second["action_results"][0]["metadata"]["side_effects"] is False
    assert second["action_results"][0]["metadata"]["original_side_effects"] is True
    assert second["action_results"][0]["output"]["remaining"] == 7
    assert second["metadata"]["action_execution_summary"]["side_effects"] is False
    assert available == 7
