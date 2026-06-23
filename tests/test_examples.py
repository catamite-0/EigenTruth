"""Tests for repository example scripts."""

import importlib
import json
import sqlite3
from types import SimpleNamespace


def test_calibrated_control_demo_defaults_to_qwen_l80_artifact_when_available():
    demo = importlib.import_module("examples.calibrated_control_demo")

    artifact = demo.default_artifact()
    diagnostics = demo.default_diagnostics_for_artifact(artifact)

    if demo.DEFAULT_QWEN_ARTIFACT_PATH.exists():
        assert artifact.model_id == "Qwen/Qwen2.5-0.5B-Instruct"
        assert artifact.target_layer == -10
        assert artifact.score_names() == ("truth_proj",)
        assert diagnostics["truth_proj"] > artifact.get_score("truth_proj").threshold
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
    for score_name in demo.default_artifact().score_names():
        assert score_name in payload["diagnostics"]
    assert payload["risk_decision"]["action"] == "abstain"
    assert payload["runtime_trace"]["summary"]["phase_counts"]["action_execution"] == 1


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
    assert payload["verification_results"] == []
    assert "initial_verification" not in {
        phase["name"] for phase in payload["runtime_trace"]["phases"]
    }
    assert stage_event["payload"]["run_verifier"] is False
    assert stage_event["payload"]["reason"] == "diagnostics and claim metadata did not require verification"
    assert payload["risk_decision"]["action"] == "accept"


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
