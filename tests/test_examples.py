"""Tests for repository example scripts."""

import importlib
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
            tool_output=None,
            request_id="test-production-tool-loop",
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
    assert statuses == ["supported", "supported", "refuted"]
    assert selected_routes == ["database_state", "tool_output_state", "tool_output_state"]
    assert route_summary["counts_by_selected_route"] == {
        "database_state": 1,
        "tool_output_state": 2,
    }
    assert route_summary["counts_by_status"] == {"supported": 2, "refuted": 1}
    assert payload["verification_results"][1]["metadata"]["actual"] == 7
    assert payload["verification_results"][2]["metadata"]["actual"] is False
    assert payload["risk_decision"]["action"] == "abstain"
    assert payload["risk_decision"]["risk_level"] == "high"
