import hashlib

from benchmarks.workflows.verification.belief_revision_kill_gate import (
    KillGatePolicy,
    evaluate_kill_gate,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _report(
    model_id: str,
    *,
    model_family: str,
    eigentruth_stubbornness: float,
    eigentruth_success: float,
):
    methods = (
        "baseline_prompt",
        "self_correction_prompt",
        "rag_evidence_only",
        "eigentruth_revision_loop",
    )
    results = [
        {
            "example_id": f"example-{index}",
            "method": method,
            "revision_answer": "answer",
            "input_prompt_sha256": _sha(f"prompt:{method}:{index}"),
            "output_sha256": _sha(f"answer:{method}:{index}"),
        }
        for method in methods
        for index in range(30)
    ]
    return {
        "schema_version": 2,
        "workflow": "belief_revision_real_model_eval",
        "model_id": model_id,
        "model_family": model_family,
        "model_revision": _sha(model_id),
        "methods": methods,
        "summary": {
            "example_count": 30,
            "by_method": {
                "baseline_prompt": {"count": 30},
                "rag_evidence_only": {"count": 30},
                "self_correction_prompt": {
                    "count": 30,
                    "stubbornness_rate": 0.4,
                    "correction_success_rate": 0.4,
                },
                "eigentruth_revision_loop": {
                    "count": 30,
                    "stubbornness_rate": eigentruth_stubbornness,
                    "correction_success_rate": eigentruth_success,
                },
            },
        },
        "results": results,
        "generation": {
            "backend": "transformers",
            "is_real_model": True,
            "model_id": model_id,
            "model_revision": _sha(model_id),
            "config": {
                "do_sample": False,
                "max_new_tokens": 64,
                "seed": 0,
                "temperature": 0.0,
                "top_p": 1.0,
            },
        },
        "dataset": {
            "split_name": "kill-test-v1",
            "example_count": 30,
            "runtime_examples_sha256": _sha("runtime"),
            "scoring_labels_sha256": _sha("labels"),
            "example_ids_sha256": _sha("example-ids"),
            "evaluation_held_out_from_prompt_development": True,
        },
        "protocol": {
            "labels_separated_from_generation_inputs": True,
            "labels_passed_to_generator": False,
            "runtime_validation_passed": True,
            "all_methods_generated": True,
        },
        "prompt_template_sha256": {method: _sha(method) for method in methods},
    }


def test_gate_continues_only_when_every_model_clears_margin() -> None:
    result = evaluate_kill_gate(
        [
            _report(
                "qwen-model",
                model_family="qwen",
                eigentruth_stubbornness=0.2,
                eigentruth_success=0.65,
            ),
            _report(
                "smollm-model",
                model_family="smollm",
                eigentruth_stubbornness=0.25,
                eigentruth_success=0.55,
            ),
        ]
    )

    assert result["decision"] == "CONTINUE_0_3"


def test_gate_pauses_when_one_model_does_not_improve() -> None:
    result = evaluate_kill_gate(
        [
            _report(
                "qwen-model",
                model_family="qwen",
                eigentruth_stubbornness=0.2,
                eigentruth_success=0.65,
            ),
            _report(
                "smollm-model",
                model_family="smollm",
                eigentruth_stubbornness=0.4,
                eigentruth_success=0.4,
            ),
        ]
    )

    assert result["decision"] == "PAUSE_PROJECT"


def test_gate_refuses_small_fixture_evidence() -> None:
    report = _report(
        "qwen-model",
        model_family="qwen",
        eigentruth_stubbornness=0.0,
        eigentruth_success=1.0,
    )
    report["summary"]["example_count"] = 4
    report["dataset"]["example_count"] = 4

    result = evaluate_kill_gate(
        [report],
        policy=KillGatePolicy(
            minimum_models=2,
            minimum_model_families=2,
            minimum_examples_per_model=20,
        ),
    )

    assert result["decision"] == "INSUFFICIENT_EVIDENCE"
    assert result["failures"]


def test_gate_refuses_fixture_or_unfingerprinted_reports() -> None:
    fixture = {
        "schema_version": 1,
        "workflow": "belief_revision_eval",
        "model_id": "fixture-open-model",
        "summary": {"example_count": 40, "by_method": {}},
    }

    result = evaluate_kill_gate(
        [fixture],
        policy=KillGatePolicy(
            minimum_models=1,
            minimum_model_families=1,
            require_qwen_and_non_qwen=False,
        ),
    )

    assert result["decision"] == "INSUFFICIENT_EVIDENCE"
    assert any("real transformers model" in failure for failure in result["failures"])


def test_gate_refuses_mismatched_dataset_fingerprints() -> None:
    qwen = _report(
        "qwen-model",
        model_family="qwen",
        eigentruth_stubbornness=0.2,
        eigentruth_success=0.65,
    )
    smollm = _report(
        "smollm-model",
        model_family="smollm",
        eigentruth_stubbornness=0.2,
        eigentruth_success=0.65,
    )
    smollm["dataset"]["runtime_examples_sha256"] = _sha("other-runtime")

    result = evaluate_kill_gate([qwen, smollm])

    assert result["decision"] == "INSUFFICIENT_EVIDENCE"
    assert any("same fingerprinted dataset" in failure for failure in result["failures"])
