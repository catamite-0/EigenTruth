import pytest

from eigentruth.adapters import (
    InMemoryWorldModelAdapter,
    RuleBasedWorldModelAdapter,
    WorldModelPrediction,
    WorldModelRule,
)
from eigentruth.control import (
    ActionExecutionStatus,
    ActionRequest,
    ActionResult,
    ControlAction,
    WorldModelActionGatePolicy,
    WorldModelActionGateStatus,
    WorldModelGuardedActionExecutor,
    audit_world_model_action_gate,
)
from eigentruth.verify import InMemoryVerifier


class RecordingExecutor:
    def __init__(self):
        self.calls = 0

    def execute(self, request, context=None):
        self.calls += 1
        return ActionResult(
            action=request.action,
            status=ActionExecutionStatus.SUCCEEDED,
            output={"reserved": True},
            metadata={"executor": type(self).__name__, "side_effects": True},
            request_id=request.request_id,
        )

    def execute_many(self, requests, context=None):
        return tuple(self.execute(request, context=context) for request in requests)


def test_world_model_action_gate_passes_supported_transition():
    world = RuleBasedWorldModelAdapter(
        rules=(
            WorldModelRule(
                name="reserve_sku",
                action_match={"type": "reserve", "sku": "sku-1"},
                conditions=({"path": "inventory.sku-1.available", "operator": "gte", "value": 1},),
                decrement_values={"inventory.sku-1.available": 1},
                confidence=0.95,
            ),
        ),
        state={"inventory": {"sku-1": {"available": 2}}},
    )
    request = ActionRequest(
        action=ControlAction.EXECUTE_TOOL,
        reason="reserve inventory",
        payload={
            "state_transition": {
                "action": {"type": "reserve", "sku": "sku-1"},
                "postcondition": {"path": "inventory.sku-1.available", "operator": "eq", "value": 1},
            }
        },
        request_id="reserve-1",
    )

    report = audit_world_model_action_gate(
        request=request,
        world_model=world,
        state={"inventory": {"sku-1": {"available": 2}}},
        policy=WorldModelActionGatePolicy(min_prediction_confidence=0.8),
    )

    assert report.status is WorldModelActionGateStatus.PASSED
    assert report.passed is True
    assert report.prediction_confidence == pytest.approx(0.95)
    assert report.summary()["postcondition_count"] == 1
    assert report.postcondition_results[0]["status"] == "supported"


def test_world_model_guarded_executor_blocks_refuted_postcondition_without_dispatch():
    world = RuleBasedWorldModelAdapter(
        rules=(
            WorldModelRule(
                name="reserve_sku",
                action_match={"type": "reserve", "sku": "sku-1"},
                decrement_values={"inventory.sku-1.available": 1},
                confidence=1.0,
            ),
        ),
        state={"inventory": {"sku-1": {"available": 2}}},
    )
    wrapped = RecordingExecutor()
    executor = WorldModelGuardedActionExecutor(
        wrapped,
        world_model=world,
        state={"inventory": {"sku-1": {"available": 2}}},
    )
    request = ActionRequest(
        action=ControlAction.EXECUTE_TOOL,
        reason="reserve inventory",
        payload={
            "state_transition": {
                "action": {"type": "reserve", "sku": "sku-1"},
                "postcondition": {"path": "inventory.sku-1.available", "operator": "eq", "value": 2},
            }
        },
        request_id="reserve-2",
    )

    result = executor.execute(request, context={"trace_id": "trace-1"})

    assert wrapped.calls == 0
    assert result.status is ActionExecutionStatus.SKIPPED
    assert result.metadata["side_effects"] is False
    assert result.metadata["world_model_gate"]["blocked"] is True
    assert result.output["world_model_gate"]["summary"]["counts_by_code"] == {"postcondition_refuted": 1}
    assert result.output["correction_request"]["action"] == "clarify"


def test_world_model_guarded_executor_dispatches_when_gate_passes():
    world = InMemoryWorldModelAdapter(verifier=InMemoryVerifier({}))
    wrapped = RecordingExecutor()
    executor = WorldModelGuardedActionExecutor(
        wrapped,
        world_model=world,
        state={"quota": {"remaining": 3}},
        policy={"min_prediction_confidence": 0.5},
    )
    request = ActionRequest(
        action=ControlAction.EXECUTE_TOOL,
        reason="use quota",
        payload={
            "state_transition": {
                "action": {"decrement": {"quota.remaining": 1}},
                "postcondition": {"path": "quota.remaining", "operator": "eq", "value": 2},
            }
        },
        request_id="quota-1",
    )

    result = executor.execute(request)

    assert wrapped.calls == 1
    assert result.status is ActionExecutionStatus.SUCCEEDED
    assert result.metadata["world_model_gate"]["passed"] is True
    assert result.metadata["world_model_guard"] == "WorldModelGuardedActionExecutor"


def test_world_model_action_gate_fails_closed_on_low_confidence_prediction():
    class LowConfidenceWorld:
        def verify(self, claim, context=None):
            raise AssertionError("not used")

        def predict(self, state, action):
            return WorldModelPrediction(
                state={"quota": {"remaining": 2}},
                confidence=0.3,
                explanation="weak prediction",
            )

        def explain(self, claim):
            return "weak"

    request = ActionRequest(
        action=ControlAction.EXECUTE_TOOL,
        reason="use quota",
        payload={
            "state_transition": {
                "action": {"type": "use_quota"},
                "postcondition": {"path": "quota.remaining", "operator": "eq", "value": 2},
            }
        },
    )

    report = audit_world_model_action_gate(
        request=request,
        world_model=LowConfidenceWorld(),
        policy={"min_prediction_confidence": 0.8},
    )

    assert report.status is WorldModelActionGateStatus.BLOCKED
    assert report.summary()["counts_by_code"]["low_prediction_confidence"] == 1


def test_world_model_action_gate_policy_parses_strict_bool_strings():
    policy = WorldModelActionGatePolicy.from_dict({
        "require_transition": "true",
        "block_on_no_rule_match": "false",
    })

    assert policy.require_transition is True
    assert policy.block_on_no_rule_match is False

    with pytest.raises(ValueError, match="require_transition"):
        WorldModelActionGatePolicy.from_dict({"require_transition": "maybe"})
