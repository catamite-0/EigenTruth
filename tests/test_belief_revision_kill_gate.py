from benchmarks.workflows.verification.belief_revision_kill_gate import (
    KillGatePolicy,
    evaluate_kill_gate,
)


def _report(model_id: str, *, eigentruth_stubbornness: float, eigentruth_success: float):
    return {
        "model_id": model_id,
        "summary": {
            "example_count": 30,
            "by_method": {
                "baseline_prompt": {},
                "rag_evidence_only": {},
                "self_correction_prompt": {
                    "stubbornness_rate": 0.4,
                    "correction_success_rate": 0.4,
                },
                "eigentruth_revision_loop": {
                    "stubbornness_rate": eigentruth_stubbornness,
                    "correction_success_rate": eigentruth_success,
                },
            },
        },
    }


def test_gate_continues_only_when_every_model_clears_margin() -> None:
    result = evaluate_kill_gate(
        [
            _report("model-a", eigentruth_stubbornness=0.2, eigentruth_success=0.65),
            _report("model-b", eigentruth_stubbornness=0.25, eigentruth_success=0.55),
        ]
    )

    assert result["decision"] == "CONTINUE_0_3"


def test_gate_pauses_when_one_model_does_not_improve() -> None:
    result = evaluate_kill_gate(
        [
            _report("model-a", eigentruth_stubbornness=0.2, eigentruth_success=0.65),
            _report("model-b", eigentruth_stubbornness=0.4, eigentruth_success=0.4),
        ]
    )

    assert result["decision"] == "PAUSE_PROJECT"


def test_gate_refuses_small_fixture_evidence() -> None:
    report = _report("model-a", eigentruth_stubbornness=0.0, eigentruth_success=1.0)
    report["summary"]["example_count"] = 4

    result = evaluate_kill_gate(
        [report],
        policy=KillGatePolicy(minimum_models=2, minimum_examples_per_model=20),
    )

    assert result["decision"] == "INSUFFICIENT_EVIDENCE"
    assert result["failures"]
