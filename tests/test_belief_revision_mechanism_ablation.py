from benchmarks.workflows.verification.belief_revision_mechanism_ablation import (
    ABLATION_ARMS,
    build_ablation_prompt,
    build_ablation_report,
    build_blind_adjudication_bundle,
    build_blind_adjudication_packet,
    evaluate_ablation_gate,
    reconcile_blind_adjudications,
)
from benchmarks.workflows.verification.belief_revision_real_model_eval import (
    CallableTextGenerator,
    RuntimeExample,
    ScoringLabel,
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


def test_ablation_matrix_exposes_only_declared_factors() -> None:
    example = _example()

    for arm in ABLATION_ARMS:
        prompt = build_ablation_prompt(example, arm.name)
        assert ("Correct City" in prompt) is arm.evidence_available
        assert ("Wrong City" in prompt) is arm.untrusted_draft_present
        assert "accepted_answers" not in prompt
        assert "expected_revision" not in prompt


def test_ablation_report_generates_six_arms_and_computes_contrasts() -> None:
    calls = []

    def generate(prompt: str, example_id: str, method: str) -> str:
        calls.append((prompt, example_id, method))
        return "Correct City" if "Evidence:" in prompt else "Wrong City"

    report = build_ablation_report(
        examples=(_example(),),
        labels=(_label(),),
        generator=CallableTextGenerator(
            generate,
            metadata={"model_revision": "test-revision"},
        ),
        model_id="test/model",
    )

    assert len(calls) == len(ABLATION_ARMS) == 6
    assert report["summary"]["result_count"] == 6
    assert report["protocol"]["labels_passed_to_generator"] is False
    assert report["summary"]["by_arm_and_case_type"]["evidence_only"][
        "contradiction"
    ]["correction_success_rate"] == 1.0
    contrast = report["contrasts"]["evidence_effect_without_explicit_stance"]
    assert contrast["correction_success_difference"] == 1.0
    assert contrast["stubbornness_difference"] == -1.0
    assert contrast["paired_outcomes"] == {
        "treatment_wins": 1,
        "ties": 0,
        "treatment_losses": 0,
    }
    assert contrast["correction_success_paired_bootstrap_95ci"] == [1.0, 1.0]


def test_blind_packet_hides_model_arm_alias_score_and_source_id() -> None:
    generator = CallableTextGenerator(
        lambda _prompt, _example_id, _method: "Correct City",
        metadata={"model_revision": "test-revision"},
    )
    report = build_ablation_report(
        examples=(_example(),),
        labels=(_label(),),
        generator=generator,
        model_id="secret/model",
    )

    packet = build_blind_adjudication_packet(
        reports=(report,),
        examples=(_example(),),
        salt="private-test-salt",
    )

    assert len(packet["items"]) == 6
    serialized = str(packet)
    assert "secret/model" not in serialized
    assert "evidence_only" not in serialized
    assert "correction_success" not in serialized
    assert "example-1" not in serialized
    assert packet["rubric"]["required_adjudicators"] == 2


def test_blind_packet_requires_private_salt() -> None:
    try:
        build_blind_adjudication_packet(reports=(), examples=(), salt="")
    except ValueError as exc:
        assert "salt" in str(exc)
    else:
        raise AssertionError("empty blinding salt should fail")


def test_two_independent_blind_annotations_reconcile_through_private_key() -> None:
    report = build_ablation_report(
        examples=(_example(),),
        labels=(_label(),),
        generator=CallableTextGenerator(
            lambda _prompt, _example_id, _method: "Correct City",
            metadata={"model_revision": "test-revision"},
        ),
        model_id="secret/model",
    )
    packet, private_key = build_blind_adjudication_bundle(
        reports=(report,),
        examples=(_example(),),
        salt="private-test-salt",
    )
    annotations = [
        {
            "adjudication_id": item["adjudication_id"],
            "semantic_verdict": "correct",
            "draft_persistence": "no",
        }
        for item in packet["items"]
    ]

    reconciliation = reconcile_blind_adjudications(
        annotation_sets=(
            {"adjudicator_id": "reviewer-a", "annotations": annotations},
            {"adjudicator_id": "reviewer-b", "annotations": annotations},
        ),
        private_key=private_key,
    )

    assert reconciliation["item_count"] == 6
    assert reconciliation["verdict_agreement_rate"] == 1.0
    assert reconciliation["unresolved_count"] == 0
    assert all(row["model_id"] == "secret/model" for row in reconciliation["results"])


def test_reconciliation_rejects_same_adjudicator_twice() -> None:
    private_key = {
        "items": [
            {
                "adjudication_id": "opaque",
                "alias_correction_success": True,
            }
        ]
    }
    annotation_set = {
        "adjudicator_id": "same-reviewer",
        "annotations": [
            {
                "adjudication_id": "opaque",
                "semantic_verdict": "correct",
                "draft_persistence": "no",
            }
        ],
    }

    try:
        reconcile_blind_adjudications(
            annotation_sets=(annotation_set, annotation_set),
            private_key=private_key,
        )
    except ValueError as exc:
        assert "independent" in str(exc)
    else:
        raise AssertionError("same adjudicator should fail")


def test_ablation_gate_pauses_distinct_loop_when_draft_hurts_one_model() -> None:
    def report(model_id: str, family: str, primary_difference: float):
        generator = CallableTextGenerator(
            lambda _prompt, _example_id, _method: "Correct City",
            metadata={
                "backend": "transformers",
                "is_real_model": True,
                "model_revision": f"{model_id}-revision",
                "config": {"temperature": 0.0},
            },
        )
        value = build_ablation_report(
            examples=(_example(),),
            labels=(_label(),),
            generator=generator,
            model_id=model_id,
            model_family=family,
            runtime_path=__file__,
            labels_path=__file__,
        )
        value["contrasts"]["draft_anchoring_with_explicit_stance"][
            "correction_success_difference"
        ] = primary_difference
        return value

    gate = evaluate_ablation_gate(
        (
            report("model-a", "family-a", -0.25),
            report("model-b", "family-b", 0.10),
        )
    )

    assert gate["decision"] == "PAUSE_DISTINCT_REVISION_LOOP"
    assert not gate["failures"]
