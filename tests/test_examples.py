"""Tests for repository example scripts."""

import importlib
import json
import sqlite3
from types import SimpleNamespace

import pytest


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
        promotion_source = payload["metadata"]["promotion_contract_source"]
        assert promotion_source.endswith("product-promotion-contract.json")
        assert any(
            version in promotion_source
            for version in ("v1_9", "v1_8", "v1_6", "v1_5", "v0_3")
        )
        assert payload["metadata"]["promotion_contract_model_id"] == "HuggingFaceTB/SmolLM2-135M-Instruct"
        assert payload["metadata"]["promotion_contract_budget_enabled"] is False
        if "v1_9" in promotion_source:
            assert payload["metadata"]["promotion_contract_recommended_runtime_seconds"] == pytest.approx(0.191662)
            assert (
                payload["metadata"]["promotion_contract_recommended_runtime_cost_source"]
                == "cache_only_total_seconds"
            )
            assert payload["metadata"]["promotion_contract_evidence_handoff_status"] == "promote"
            assert payload["metadata"]["promotion_contract_evidence_handoff_present_metric_count"] == 65
            assert payload["metadata"]["promotion_contract_evidence_handoff_missing_metric_count"] == 0
            assert payload["metadata"]["promotion_contract_evidence_handoff_blocked_group_count"] == 0
            assert payload["metadata"]["promotion_contract_evidence_handoff_group_statuses"][
                "frontier_release_evidence"
            ] == "promote"
            assert (
                payload["metadata"]["promotion_contract_product_runtime_drift_status"]
                == "promote"
            )
            assert (
                payload["metadata"]["promotion_contract_product_runtime_drift_compared_metric_count"]
                == 107
            )
            assert (
                payload["metadata"]["promotion_contract_product_runtime_drift_blocked_metric_count"]
                == 0
            )
        assert payload["metadata"]["promotion_contract_metadata"]["recommended_selector_replay_candidate"] == "default"
        assert payload["metadata"]["promotion_contract_metadata"]["selector_replay_status"] == "promote"
        assert payload["metadata"]["promotion_contract_metadata"]["product_runtime_drift_status"] == "promote"
        assert payload["metadata"]["promotion_contract_metadata"]["product_runtime_drift_blocked_metric_count"] == 0
        assert payload["metadata"]["promotion_contract_metadata"]["adapter_family_required_routes"]
        assert payload["metadata"]["promotion_contract_metadata"]["required_route_baseline_routes"]
    for score_name in demo.default_artifact().score_names():
        assert score_name in payload["diagnostics"]
    assert payload["risk_decision"]["action"] == "abstain"
    assert payload["final_answer"]["status"] == "abstained"
    assert payload["final_answer"]["answerable"] is False
    assert payload["metadata"]["final_answer_summary"]["status"] == "abstained"
    assert payload["runtime_trace"]["summary"]["phase_counts"]["action_execution"] == 1


def test_calibrated_control_demo_can_use_multiple_testing_gate(tmp_path):
    demo = importlib.import_module("examples.calibrated_control_demo")
    eval_conformal = importlib.import_module("benchmarks.eval_conformal")
    from eigentruth.calibration import (
        CalibrationArtifact,
        CalibrationScore,
    )

    artifact_path = tmp_path / "calibration.json"
    gate_path = tmp_path / "multiple-testing-calibration.json"
    report_path = tmp_path / "multiple-testing-report.json"
    scores_path = tmp_path / "scores.json"
    CalibrationArtifact(
        model_id="demo-model",
        target_layer=-1,
        scores=(CalibrationScore("maha_last", threshold=1000.0),),
        eigentruth_version="0.1.0",
    ).save_json(artifact_path)
    scores_path.write_text(
        json.dumps({
            "config": {"model": "demo-model", "layer": -1},
            "labels": [0] * 20 + [1, 1, 1, 1],
            "scores": {
                "support_score": list(range(100, 120)) + [0, 1, 115, 116],
                "maha_last": list(range(20)) + [100, 101, 5, 6],
            },
        }),
        encoding="utf-8",
    )
    eval_conformal.run(
        SimpleNamespace(
            scores=str(scores_path),
            signal="support_score",
            signals=None,
            repeats=1,
            seed=0,
            json=None,
            save_calibration=None,
            save_adaptive_calibration=None,
            save_abstention_report=None,
            include_abstention_report=False,
            save_abstention_comparison=None,
            include_abstention_comparison=False,
            save_abstention_release_gate=None,
            include_abstention_release_gate=False,
            save_multiple_testing_report=str(report_path),
            save_multiple_testing_calibration=str(gate_path),
            include_multiple_testing_report=False,
            multiple_testing_signals="support_score,maha_last",
            multiple_testing_alpha=0.30,
            multiple_testing_method="by",
            save_sweep_report=None,
            save_best_calibration=None,
            best_by="auroc",
            artifact_alpha=0.10,
            abstention_alpha=0.10,
            abstention_signal=None,
            abstention_direction=None,
            abstention_signals=None,
            abstention_best_by="conditional_correctness_lower_bound",
            min_abstention_conditional_correctness_lower_bound=0.8,
            max_abstention_rate=0.5,
            direction="lower",
            adaptive_feature=(),
            adaptive_feature_weight=(),
            adaptive_intercept=0.0,
            adaptive_score_name="adaptive",
            confidence_signal="nll_answer",
            confidence_direction=None,
            confidence_top_fraction=0.25,
            disable_confidence_audit=True,
            model_id=None,
            model_revision=None,
            target_layer=None,
            created_at="2026-06-29T00:00:00+00:00",
            commit_sha=None,
            artifact_manifest=None,
        )
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["config"]["directions"] == {
        "support_score": "lower",
        "maha_last": "higher",
    }
    assert gate_path.exists()

    payload = demo.run(
        SimpleNamespace(
            artifact=str(artifact_path),
            multiple_testing_gate=str(gate_path),
            diagnostics='{"maha_last": 100.0, "support_score": 0.0}',
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
            request_id="multiple-testing-gate-demo",
            output=None,
            registry=None,
        )
    )
    gate_trace = payload["risk_decision"]["diagnostics"]["multiple_testing_gate"]

    assert payload["risk_decision"]["action"] == "abstain"
    assert payload["final_answer"]["status"] == "abstained"
    assert payload["metadata"]["multiple_testing_gate_enabled"] is True
    assert payload["metadata"]["multiple_testing_gate_source"] == str(gate_path)
    assert payload["metadata"]["multiple_testing_gate_signals"] == ("support_score", "maha_last")
    assert gate_trace["status"] == "rejected"
    assert set(gate_trace["rejected_signal_names"]) == {"support_score", "maha_last"}


def test_calibrated_control_demo_preserves_bool_diagnostics_for_fail_closed(tmp_path):
    demo = importlib.import_module("examples.calibrated_control_demo")
    from eigentruth.calibration import CalibrationArtifact, CalibrationScore

    artifact_path = tmp_path / "calibration.json"
    CalibrationArtifact(
        model_id="demo-model",
        target_layer=-1,
        scores=(CalibrationScore("maha_last", threshold=1.0),),
        eigentruth_version="0.1.0",
    ).save_json(artifact_path)

    payload = demo.run(
        SimpleNamespace(
            artifact=str(artifact_path),
            multiple_testing_gate=None,
            sequential_gate=None,
            diagnostics='{"maha_last": true}',
            diagnostics_sequence=None,
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
            request_id="bool-diagnostic-demo",
            output=None,
            registry=None,
        )
    )

    assert payload["risk_decision"]["action"] == "clarify"
    assert payload["risk_decision"]["risk_level"] == "unknown"
    assert payload["risk_decision"]["diagnostics"]["invalid_scores"] == ("maha_last",)


def test_calibrated_control_demo_rejects_bool_runtime_budget_mapping():
    demo = importlib.import_module("examples.calibrated_control_demo")

    with pytest.raises(ValueError, match="max_phase_seconds.initial_verification"):
        demo.runtime_budget_policy_from_args(
            SimpleNamespace(
                promotion_contract=None,
                max_runtime_phase_seconds='{"initial_verification": true}',
            )
        )


def test_calibrated_control_demo_can_replay_sequential_gate_trace(tmp_path):
    demo = importlib.import_module("examples.calibrated_control_demo")
    from eigentruth.calibration import (
        CalibrationArtifact,
        CalibrationScore,
        SequentialConformalCalibrator,
    )

    artifact_path = tmp_path / "calibration.json"
    gate_path = tmp_path / "sequential-calibration.json"
    output_path = tmp_path / "sequence-trace.json"
    CalibrationArtifact(
        model_id="demo-model",
        target_layer=-1,
        scores=(CalibrationScore("maha_last", threshold=1000.0),),
        eigentruth_version="0.1.0",
    ).save_json(artifact_path)
    SequentialConformalCalibrator(alpha=0.5, schedule="linear").calibrate(
        model_id="demo-model",
        target_layer=-1,
        signal_name="support_score",
        calibration_scores=[10.0, 11.0, 12.0, 13.0],
        direction="lower",
        eigentruth_version="0.1.0",
    ).save_json(gate_path)

    payload = demo.run(
        SimpleNamespace(
            artifact=str(artifact_path),
            multiple_testing_gate=None,
            sequential_gate=str(gate_path),
            diagnostics=None,
            diagnostics_sequence=json.dumps((
                {"maha_last": 1.0, "support_score": 9.0},
                {"maha_last": 1.0, "support_score": 12.0},
            )),
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
            request_id="sequential-gate-demo",
            output=str(output_path),
            registry=None,
            compact_json=False,
        )
    )
    written = json.loads(output_path.read_text(encoding="utf-8"))
    first_gate = payload["risk_decisions"][0]["diagnostics"]["sequential_gate"]
    second_gate = payload["risk_decisions"][1]["diagnostics"]["sequential_gate"]

    assert written["trace_format"] == "risk_decision_sequence"
    assert payload["trace_format"] == "risk_decision_sequence"
    assert payload["metadata"]["sequential_gate_enabled"] is True
    assert payload["metadata"]["sequential_gate_source"] == str(gate_path)
    assert payload["metadata"]["sequence_decision_summary"]["action_counts"] == {
        "abstain": 1,
        "accept": 1,
    }
    assert payload["risk_decisions"][0]["action"] == "abstain"
    assert payload["risk_decisions"][1]["action"] == "accept"
    assert first_gate["status"] == "rejected"
    assert first_gate["step"] == 1
    assert first_gate["report_summary"]["rejected_steps"] == (1,)
    assert second_gate["status"] == "passed"
    assert payload["events"][0]["event_type"] == "sequence_risk_decision"


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
    assert payload["final_answer"]["status"] == "abstained"
    assert payload["summaries"]["final_answer"]["status"] == "abstained"
    assert payload["metadata"]["final_answer_summary"]["available"] is True
    assert payload["runtime_trace"] is None
    assert payload["truncation"]["runtime_trace_included"] is False
    assert payload["summaries"]["runtime"]["measured_phases"] > 0
    assert payload["truncation"]["claims"]["included"] <= 1
    assert payload["truncation"]["verification_results"]["included"] <= 1
    assert payload["metadata"]["artifact_source"] == demo.artifact_source(None)
    if demo.default_promotion_contract_path() is not None:
        assert payload["metadata"]["promotion_contract_source"].endswith("product-promotion-contract.json")


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


def test_calibrated_control_demo_can_route_on_learned_pre_generation_risk(tmp_path):
    demo = importlib.import_module("examples.calibrated_control_demo")
    artifact = demo.default_artifact()
    policy_path = tmp_path / "pre-generation-policy.json"
    learned_risk_path = tmp_path / "learned-risk.json"
    policy_path.write_text(
        json.dumps({"route_on_learned_risk": True, "soft_risk_config": None}),
        encoding="utf-8",
    )
    learned_risk_path.write_text(
        json.dumps({
            "score": 2.0,
            "probability": 0.88,
            "risk_level": "high",
            "source": "unit_pre_generation_probe",
            "layer_idx": 2,
        }),
        encoding="utf-8",
    )

    payload = demo.run(
        SimpleNamespace(
            artifact=None,
            diagnostics=json.dumps(demo.low_diagnostics_for_artifact(artifact)),
            text="Explain calibration intuitively.",
            facts=None,
            evidence=None,
            refutations=None,
            retrieval_evidence=None,
            enable_calculator=False,
            calculator_context=None,
            runtime_profile=None,
            runtime_profile_selector_policy=None,
            pre_generation_profile="auto",
            pre_generation_risk_policy=str(policy_path),
            pre_generation_metadata=None,
            pre_generation_learned_risk=str(learned_risk_path),
            staged_verification=None,
            runtime_trace=True,
            request_id="pre-generation-learned-risk",
            output=None,
            registry=None,
        )
    )

    assessment = payload["metadata"]["pre_generation_risk_assessment"]

    assert payload["metadata"]["runtime_profile"] == "audit"
    assert payload["metadata"]["runtime_profile_source"] == "pre_generation"
    assert payload["metadata"]["pre_generation_risk_policy"]["route_on_learned_risk"] is True
    assert assessment["selected_profile"] == "audit"
    assert assessment["reason"] == "learned pre-generation risk estimate exceeded high threshold"
    assert assessment["learned_risk"]["source"] == "unit_pre_generation_probe"
    assert assessment["learned_risk"]["layer_idx"] == 2


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


def test_calibrated_control_demo_records_selfcheck_promotion_evidence(tmp_path):
    demo = importlib.import_module("examples.calibrated_control_demo")
    from eigentruth.control import ProductPromotionContract
    from eigentruth.registry import ArtifactRegistry, build_artifact_manifest

    contract_path = tmp_path / "promotion-contract.json"
    selfcheck_dir = tmp_path / "selfcheck"
    selfcheck_dir.mkdir()
    selfcheck_report_path = selfcheck_dir / "workflow.json"
    selfcheck_manifest_path = selfcheck_dir / "artifact-manifest.json"
    selfcheck_registry_path = selfcheck_dir / "registry.json"
    selfcheck_report_path.write_text(
        json.dumps({
            "workflow": "selfcheck_signal_fusion_workflow",
            "status": "promote",
            "sample_quality": {"status": "pass", "passed": True},
            "fusion_summary": {"runs": [{"name": "tiny", "auroc": 0.7}]},
        }),
        encoding="utf-8",
    )
    selfcheck_manifest_path.write_text(
        json.dumps(
            build_artifact_manifest(
                {"selfcheck_signal_fusion_workflow": selfcheck_report_path},
                root=selfcheck_dir,
                metadata={"workflow": "selfcheck_signal_fusion_workflow"},
            )
        ),
        encoding="utf-8",
    )
    ArtifactRegistry.load_json(selfcheck_registry_path).record_report(
        name="selfcheck-signal-fusion-workflow",
        path=selfcheck_report_path,
        version="0.1",
        metadata={"artifact_manifest": str(selfcheck_manifest_path)},
    ).save_json()
    ProductPromotionContract(
        model_id="demo-model",
        runtime={"layer": -1},
        verifier_route={"route": "fallback"},
        source_status="promote",
        selfcheck_signal_fusion_workflow={
            "report_path": "selfcheck/workflow.json",
            "manifest_path": "selfcheck/artifact-manifest.json",
            "registry": "selfcheck/registry.json",
            "record_key": "report:selfcheck-signal-fusion-workflow:0.1",
            "status": "promote",
            "sample_quality_status": "pass",
            "sample_quality_passed": True,
            "fusion_run_count": 1,
        },
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
            enable_calculator=False,
            calculator_context=None,
            runtime_profile=None,
            staged_verification=None,
            runtime_trace=True,
            promotion_contract=str(contract_path),
            verify_selfcheck_signal_fusion_manifest=True,
            include_selfcheck_signal_fusion_record=True,
            require_selfcheck_signal_fusion_manifest_verification=True,
            require_selfcheck_signal_fusion_record=True,
            request_id="test-selfcheck-promotion-evidence",
            output=None,
            registry=None,
        )
    )

    metadata = payload["metadata"]
    assert metadata["selfcheck_signal_fusion_workflow_report"] == str(selfcheck_report_path)
    assert metadata["selfcheck_signal_fusion_workflow_manifest"] == str(
        selfcheck_manifest_path
    )
    assert metadata["selfcheck_signal_fusion_workflow_manifest_verification"]["passed"] is True
    assert metadata["selfcheck_signal_fusion_workflow_manifest_verification"]["checked"] == 1
    assert metadata["selfcheck_signal_fusion_workflow_registry"] == str(
        selfcheck_registry_path
    )
    assert metadata["selfcheck_signal_fusion_workflow_registry_key"] == (
        "report:selfcheck-signal-fusion-workflow:0.1"
    )
    assert metadata["selfcheck_signal_fusion_workflow_registry_record"]["metadata"] == {
        "artifact_manifest": str(selfcheck_manifest_path)
    }
    assert metadata["selfcheck_signal_fusion_workflow_sample_quality_passed"] is True
    assert metadata["selfcheck_signal_fusion_workflow_fusion_run_count"] == 1
    gate = metadata["selfcheck_signal_fusion_evidence_gate"]
    assert gate["enabled"] is True
    assert gate["passed"] is True
    assert gate["policy"]["require_manifest_verification"] is True
    assert gate["policy"]["require_registry_record"] is True
    assert gate["failures"] == []


def test_calibrated_control_demo_fails_selfcheck_evidence_gate(tmp_path):
    demo = importlib.import_module("examples.calibrated_control_demo")
    from eigentruth.control import ProductPromotionContract

    contract_path = tmp_path / "promotion-contract.json"
    ProductPromotionContract(
        model_id="demo-model",
        runtime={"layer": -1},
        verifier_route={"route": "fallback"},
        source_status="promote",
        selfcheck_signal_fusion_workflow={
            "status": "blocked",
            "sample_quality_status": "fail",
            "sample_quality_passed": False,
            "fusion_run_count": 0,
        },
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
            enable_calculator=False,
            calculator_context=None,
            runtime_profile=None,
            staged_verification=None,
            runtime_trace=True,
            promotion_contract=str(contract_path),
            require_selfcheck_signal_fusion_evidence=True,
            request_id="test-selfcheck-evidence-gate-fail",
            output=None,
            registry=None,
        )
    )

    gate = payload["metadata"]["selfcheck_signal_fusion_evidence_gate"]
    failure_metrics = {failure["metric"] for failure in gate["failures"]}

    assert gate["enabled"] is True
    assert gate["passed"] is False
    assert failure_metrics == {
        "selfcheck_signal_fusion_workflow_status",
        "selfcheck_signal_fusion_workflow_sample_quality_passed",
        "selfcheck_signal_fusion_workflow_report",
        "selfcheck_signal_fusion_workflow_manifest",
        "selfcheck_signal_fusion_workflow_fusion_run_count",
    }
    assert gate["checks"][0]["value"] == "blocked"


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


def test_calibrated_control_demo_can_enforce_claim_coherence():
    demo = importlib.import_module("examples.calibrated_control_demo")

    payload = demo.run(
        SimpleNamespace(
            artifact=None,
            diagnostics='{"truth_proj": 0.0}',
            text="The trial was randomized. Therefore the treatment is proven effective.",
            facts='{"Therefore the treatment is proven effective": "supported"}',
            evidence=None,
            refutations=None,
            retrieval_evidence=None,
            enable_calculator=False,
            calculator_context=None,
            runtime_profile="audit",
            staged_verification=None,
            runtime_trace=True,
            enforce_claim_coherence=True,
            request_id="test-claim-coherence-demo",
            output=None,
            registry=None,
        )
    )

    coherence = payload["metadata"]["claim_coherence"]
    event_types = {event["event_type"] for event in payload["events"]}

    assert payload["metadata"]["claim_coherence_requested"] is True
    assert coherence["enabled"] is True
    assert coherence["dependency_count"] == 1
    assert coherence["blocked_claim_ids"] == ("c2",)
    assert payload["verification_results"][1]["status"] == "insufficient_evidence"
    assert payload["verification_results"][1]["metadata"]["claim_coherence"] == {
        "blocked": True,
        "original_status": "supported",
        "parent_id": "c1",
        "parent_status": "insufficient_evidence",
        "relation": "discourse_marker",
    }
    assert payload["risk_decision"]["action"] == "retrieve"
    assert payload["final_answer"]["status"] == "needs_retrieval"
    assert "initial_claim_coherence" in event_types


def test_calibrated_control_demo_can_use_default_frontier_audit_contract_evidence():
    demo = importlib.import_module("examples.calibrated_control_demo")
    contract_path = demo.default_promotion_contract_path()
    if contract_path is None:
        pytest.skip("frontier-audit promotion contract artifact is not available")
    if "smollm2_product_promotion_contract_v1_9" not in str(contract_path):
        pytest.skip("frontier-audit v1.9 promotion contract artifact is not available")
    assert "smollm2_product_promotion_contract_v1_9" in str(contract_path)
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
            promotion_contract_evidence_handoff_manifest=None,
            promotion_contract_registry="artifacts/local-release-registry.json",
            promotion_contract_registry_key=None,
            verify_promotion_contract_manifest=True,
            verify_promotion_contract_evidence_handoff_manifest=True,
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
        "product_promotion_contract:smollm2-product-promotion-contract:1.9"
    )
    assert payload["metadata"]["promotion_contract_recommended_runtime_seconds"] == pytest.approx(0.191662)
    assert payload["metadata"]["promotion_contract_recommended_runtime_cost_source"] == "cache_only_total_seconds"
    assert payload["metadata"]["promotion_contract_evidence_handoff_manifest"].endswith(
        "evidence-handoff-artifact-manifest.json"
    )
    assert payload["metadata"]["promotion_contract_evidence_handoff_manifest_verification"]["passed"] is True
    assert payload["metadata"]["promotion_contract_evidence_handoff_manifest_verification"]["checked"] == 11
    assert payload["metadata"]["promotion_contract_evidence_handoff_status"] == "promote"
    assert payload["metadata"]["promotion_contract_evidence_handoff_present_metric_count"] == 65
    assert payload["metadata"]["promotion_contract_evidence_handoff_missing_metric_count"] == 0
    assert payload["metadata"]["promotion_contract_evidence_handoff_blocked_group_count"] == 0
    assert payload["metadata"]["promotion_contract_evidence_handoff_group_statuses"] == {
        "action_gate": "promote",
        "counterfactual": "promote",
        "covered_fact_property": "promote",
        "frontier_release_evidence": "promote",
        "pre_generation": "promote",
        "promotion": "promote",
        "triple_audit": "promote",
    }
    assert payload["metadata"]["promotion_contract_frontier_release_evidence"][
        "decision_status"
    ] == "promote"
    assert (
        payload["metadata"]["promotion_contract_product_runtime_drift_status"]
        == "promote"
    )
    assert (
        payload["metadata"]["promotion_contract_product_runtime_drift_compared_metric_count"]
        == 107
    )
    assert (
        payload["metadata"]["promotion_contract_product_runtime_drift_blocked_metric_count"]
        == 0
    )
    assert payload["metadata"]["promotion_contract_metadata"]["recommended_performance_baseline_record"] == (
        "performance_baseline:smollm2-l8-read-cache-worker-sweep-score-fusion-performance-baseline:0.2"
    )
    assert payload["metadata"]["promotion_contract_metadata"]["recommended_selector_replay_candidate"] == "default"
    assert payload["metadata"]["promotion_contract_metadata"]["selector_replay_status"] == "promote"
    assert payload["metadata"]["promotion_contract_metadata"]["product_runtime_drift_status"] == "promote"
    assert payload["metadata"]["promotion_contract_metadata"]["product_runtime_drift_blocked_metric_count"] == 0
    assert payload["metadata"]["promotion_contract_metadata"]["adapter_family_promotion_status"] == "promote"
    assert payload["metadata"]["promotion_contract_metadata"]["adapter_family_required_routes"] == [
        "structured_state",
        "state_transition",
        "triple_evidence",
    ]
    assert payload["metadata"]["promotion_contract_metadata"]["required_route_baseline_status"] == "promote"
    assert payload["metadata"]["promotion_contract_metadata"]["required_route_baseline_records"] == [
        "benchmark_manifest:smollm2-l80-retrieval-structured-qa-route:0.6",
        "benchmark_manifest:wikidata-country-core-facts-structured-fact-canonical-route:0.1",
        "benchmark_manifest:wikidata-country-core-facts-structured-fact-paraphrase-route:0.1",
    ]
    assert payload["metadata"]["promotion_contract_metadata"]["required_route_baseline_routes"] == [
        "retrieval_structured_qa",
        "structured_fact",
        "structured_fact",
    ]
    assert payload["metadata"]["promotion_contract_metadata"]["required_route_budget_policy"][
        "required_route_min_selected"
    ] == 200.0
    assert payload["verification_results"][0]["metadata"]["selected_route"] == "structured_qa"
    assert route_summary["mean_attempted_route_count"] == 1.0
    assert runtime_budget["passed"] is True
    assert runtime_budget["enabled"] is True
    assert runtime_budget["policy"]["max_mean_attempted_route_count"] == 3.0
    assert runtime_budget["policy"]["max_retrieval_use_rate"] == 1.0


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
            min_world_model_confidence=0.9,
            request_id="test-state-transition-demo",
            output=None,
        )
    )

    statuses = [result["status"] for result in payload["verification_results"]]
    routes = [result["metadata"]["decision_rule"] for result in payload["verification_results"]]

    assert payload["metadata"]["verifier_type"] == "StateTransitionVerifier"
    assert payload["metadata"]["world_model_type"] == "InMemoryWorldModelAdapter"
    assert payload["metadata"]["min_world_model_confidence"] == pytest.approx(0.9)
    assert payload["metadata"]["business_domain"] == "order_fulfillment_transition"
    assert payload["diagnostics"] == {"truth_proj": 0.0}
    assert statuses == ["supported", "refuted"]
    assert routes == ["transition_postcondition_passed", "transition_postcondition_failed"]
    assert payload["risk_decision"]["action"] == "abstain"
    assert payload["risk_decision"]["risk_level"] == "high"

    with pytest.raises(ValueError, match="min-world-model-confidence"):
        demo.run(
            SimpleNamespace(
                diagnostics=None,
                state=None,
                min_world_model_confidence=1.1,
                request_id="invalid-state-transition-demo",
                output=None,
            )
        )


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
