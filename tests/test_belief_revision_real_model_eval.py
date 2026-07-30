import pytest

from benchmarks.workflows.verification.belief_revision_real_model_eval import (
    CallableTextGenerator,
    RuntimeExample,
    ScoringLabel,
    build_method_prompt,
    build_real_model_report,
    score_generated_answer,
)
from benchmarks.workflows.verification.build_belief_revision_kill_test import (
    DEFAULT_SOURCE,
    FORBIDDEN_RUNTIME_FIELDS,
    _find_forbidden_fields,
    build_kill_test_rows,
)


def _example() -> RuntimeExample:
    return RuntimeExample(
        example_id="example-1",
        prompt="What is the capital of Exampleland?",
        initial_answer="Wrong City",
        claims=("Wrong City is the capital of Exampleland.",),
        evidence_docs=(
            {
                "record_id": "e1",
                "evidence_text": "The capital of Exampleland is Correct City.",
                "source": "source:1",
            },
        ),
    )


def _label() -> ScoringLabel:
    return ScoringLabel(
        example_id="example-1",
        case_type="contradiction",
        expected_action="revise",
        expected_revision="Correct City",
        accepted_answers=("Correct City",),
        rejected_answers=("Wrong City",),
        risk_category="entity_conflict",
    )


def test_kill_test_builder_separates_48_runtime_rows_from_labels() -> None:
    runtime_rows, labels, report = build_kill_test_rows(DEFAULT_SOURCE)

    assert len(runtime_rows) == len(labels) == 48
    assert report["summary"]["case_counts"] == {
        "contradiction": 36,
        "insufficient": 6,
        "support": 6,
    }
    assert not any(_find_forbidden_fields(row) for row in runtime_rows)
    assert all(
        field not in runtime_rows[0]
        for field in ("expected_revision", "accepted_answers", "rejected_answers")
    )
    assert labels[0]["expected_revision"]


def test_runtime_example_rejects_scoring_fields_recursively() -> None:
    payload = {
        "example_id": "example-1",
        "prompt": "Question?",
        "initial_answer": "Draft.",
        "claims": ["Draft."],
        "evidence_docs": [
            {
                "record_id": "e1",
                "evidence_text": "Evidence.",
                "corrected_claim": "Answer.",
            }
        ],
    }

    with pytest.raises(ValueError, match="corrected_claim"):
        RuntimeExample.from_dict(payload)

    assert "corrected_claim" in FORBIDDEN_RUNTIME_FIELDS


def test_method_prompts_expose_only_the_intended_runtime_fields() -> None:
    example = _example()

    baseline = build_method_prompt(example, "baseline_prompt")
    self_correction = build_method_prompt(example, "self_correction_prompt")
    evidence_only = build_method_prompt(example, "rag_evidence_only")
    eigentruth = build_method_prompt(example, "eigentruth_revision_loop")

    assert "Wrong City" in baseline
    assert "Correct City" not in baseline
    assert "Evidence:" not in self_correction
    assert "Correct City" in evidence_only
    assert "Wrong City" not in evidence_only
    assert "Correct City" in eigentruth
    assert "Wrong City" in eigentruth
    for prompt in (baseline, self_correction, evidence_only, eigentruth):
        assert "expected_revision" not in prompt
        assert "corrected_claim" not in prompt
        assert "accepted_answers" not in prompt


def test_real_model_report_generates_before_scoring_and_records_fingerprints() -> None:
    calls = []

    def generate(prompt: str, example_id: str, method: str) -> str:
        calls.append((prompt, example_id, method))
        return "Correct City" if "Evidence:" in prompt else "Wrong City"

    generator = CallableTextGenerator(
        generate,
        metadata={"model_revision": "test-revision"},
    )
    report = build_real_model_report(
        examples=(_example(),),
        labels=(_label(),),
        generator=generator,
        model_id="test/model",
    )

    assert len(calls) == 4
    assert report["schema_version"] == 2
    assert report["protocol"]["labels_passed_to_generator"] is False
    assert report["summary"]["result_count"] == 4
    assert all(len(row["input_prompt_sha256"]) == 64 for row in report["results"])
    assert report["summary"]["by_method"]["eigentruth_revision_loop"][
        "correction_success_rate"
    ] == 1.0
    assert report["summary"]["by_method"]["self_correction_prompt"][
        "stubbornness_rate"
    ] == 1.0


def test_scoring_rejects_answers_that_repeat_both_wrong_and_correct_claims() -> None:
    score = score_generated_answer(
        answer="Wrong City was proposed, but Correct City is also mentioned.",
        initial_answer="Wrong City",
        label=_label(),
    )

    assert score["correction_success"] is False
    assert score["unsupported_persistence"] is True
    assert score["stubbornness"] is True
