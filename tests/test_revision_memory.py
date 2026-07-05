import json

from eigentruth.control import ProductTrace
from eigentruth.inference_control import (
    CorrectionPolicy,
    InferenceCorrectionController,
    RevisionAction,
)
from eigentruth.memory import CorrectionBuffer, CorrectionRecord, EvidenceRecord, TruthMemory
from eigentruth.registry import load_and_verify_artifact_manifest
from eigentruth.revision import (
    BeliefRevisionExample,
    EvidenceGroundedRevisionEngine,
    RevisionTrace,
    evaluate_belief_revision_example,
    revision_metadata,
)


def test_truth_memory_search_and_jsonl_roundtrip(tmp_path):
    record = EvidenceRecord(
        record_id="e1",
        claim="长城位于上海",
        evidence_text="长城主要位于中国北方，不在上海。",
        stance="contradict",
        corrected_claim="长城主要位于中国北方，不在上海。",
    )
    memory = TruthMemory((record,))

    assert memory.search("长城位于上海", stance="contradict")[0].record_id == "e1"

    path = tmp_path / "memory.jsonl"
    memory.write_jsonl(path)
    loaded = TruthMemory.load_jsonl(path)
    assert loaded.records == memory.records


def test_correction_buffer_exports_only_verified_successes(tmp_path):
    evidence = EvidenceRecord(
        record_id="e1",
        claim="水在标准大气压下50摄氏度沸腾",
        evidence_text="标准大气压下水的沸点约为100摄氏度。",
        stance="contradict",
        corrected_claim="水在标准大气压下约100摄氏度沸腾。",
    )
    verified = CorrectionRecord(
        record_id="r1",
        prompt="水在标准大气压下多少度沸腾？",
        initial_answer="水在标准大气压下50摄氏度沸腾。",
        revised_answer="水在标准大气压下约100摄氏度沸腾。",
        claims=("水在标准大气压下50摄氏度沸腾",),
        evidence_records=(evidence,),
        verifier_status="verified",
        correction_success=True,
    )
    unverified = CorrectionRecord(
        record_id="r2",
        prompt="水在标准大气压下多少度沸腾？",
        initial_answer="水在标准大气压下50摄氏度沸腾。",
        revised_answer="水在标准大气压下约100摄氏度沸腾。",
        verifier_status="unverified",
        correction_success=True,
    )
    failed = CorrectionRecord(
        record_id="r3",
        prompt="水在标准大气压下多少度沸腾？",
        initial_answer="水在标准大气压下50摄氏度沸腾。",
        revised_answer="水在标准大气压下50摄氏度沸腾。",
        verifier_status="verified",
        correction_success=False,
    )
    buffer = CorrectionBuffer((verified, unverified, failed))

    assert buffer.verified_records() == (verified,)
    assert len(buffer.training_records(format="sft")) == 1
    assert len(buffer.training_records(format="dpo")) == 1

    buffer_path = tmp_path / "buffer.jsonl"
    buffer.write_jsonl(buffer_path)
    assert CorrectionBuffer.load_jsonl(buffer_path).verified_records()[0].record_id == "r1"


def test_belief_revision_schema_engine_and_product_trace_metadata():
    example = BeliefRevisionExample(
        prompt="鲁迅的原名是不是周树人？",
        initial_answer="鲁迅的原名是周作人。",
        claims=("鲁迅的原名是周作人",),
        evidence_docs=(
            EvidenceRecord(
                record_id="e1",
                claim="鲁迅的原名是周作人",
                evidence_text="鲁迅原名周树人。周作人是鲁迅的弟弟。",
                stance="contradict",
                corrected_claim="鲁迅的原名是周树人。",
            ),
        ),
        contradiction_label=True,
        expected_revision="鲁迅的原名是周树人。",
    )
    payload = example.to_dict()
    assert BeliefRevisionExample.from_dict(payload).claims == example.claims

    result = evaluate_belief_revision_example(
        example,
        model_id="fixture-open-model",
        engine=EvidenceGroundedRevisionEngine(),
    )

    assert result.revision_answer == "鲁迅的原名是周树人。"
    assert result.correction_success is True
    assert result.stubbornness is False
    assert result.unsupported_persistence is False

    trace = RevisionTrace.from_dict(result.revision_trace)
    product_trace = ProductTrace(metadata=revision_metadata(trace))
    bounded = product_trace.to_bounded_dict()
    assert bounded["metadata"]["revision"]["summary"]["status_counts"]["contradicted"] == 1


def test_revision_engine_does_not_rewrite_supported_or_insufficient_claims():
    engine = EvidenceGroundedRevisionEngine()
    supported = engine.revise(
        prompt="巴黎是不是法国首都？",
        initial_answer="巴黎是法国首都。",
        claims=("巴黎是法国首都",),
        evidence_records=(
            EvidenceRecord(
                record_id="e1",
                claim="巴黎是法国首都",
                evidence_text="巴黎是法国首都。",
                stance="support",
            ),
        ),
    )
    insufficient = engine.revise(
        prompt="某产品是否已经发布？",
        initial_answer="某产品已经发布。",
        claims=("某产品已经发布",),
        evidence_records=(
            EvidenceRecord(
                record_id="e2",
                claim="某产品已经发布",
                evidence_text="没有找到可验证的发布时间。",
                stance="neutral",
            ),
        ),
    )

    assert supported.action == "accept"
    assert supported.revised_answer == "巴黎是法国首都。"
    assert insufficient.action == "retrieve_more"
    assert "cannot resolve" in insufficient.revised_answer


def test_inference_correction_controller_prioritizes_revision_signals():
    controller = InferenceCorrectionController(CorrectionPolicy(risk_score_threshold=0.8))

    decision = controller.decide(
        revision_trace={
            "summary": {"status_counts": {"contradicted": 1}},
            "unsupported_persistence": False,
        },
        diagnostics={"risk_score": 0.1},
    )
    assert decision.action is RevisionAction.REVISE

    high_risk = controller.decide(diagnostics={"risk_score": 0.95})
    assert high_risk.action is RevisionAction.REGENERATE

    clean = controller.decide()
    assert clean.action is RevisionAction.ACCEPT


def test_belief_revision_baseline_manifest_verifies():
    verification = load_and_verify_artifact_manifest(
        "artifacts/baselines/belief_revision_text/artifact-manifest.json",
        root=".",
    )
    assert verification.passed, verification.to_dict()
    assert verification.checked == 1


def test_belief_revision_workflow_and_training_export(tmp_path):
    from benchmarks.workflows.calibration.correction_training_export import (
        export_correction_training_data,
    )
    from benchmarks.workflows.verification.belief_revision_eval import (
        DEFAULT_EXAMPLES,
        load_belief_revision_examples,
        write_belief_revision_report,
    )

    report_path = tmp_path / "belief-revision-report.json"
    manifest_path = tmp_path / "artifact-manifest.json"
    report = write_belief_revision_report(
        examples_path=DEFAULT_EXAMPLES,
        model_id="fixture-open-model",
        json_path=report_path,
        artifact_manifest_path=manifest_path,
    )
    summary = report["summary"]["by_method"]
    assert len(load_belief_revision_examples(DEFAULT_EXAMPLES)) == 4
    assert summary["eigentruth_revision_loop"]["stubbornness_rate"] < summary["baseline_prompt"][
        "stubbornness_rate"
    ]
    assert summary["eigentruth_revision_loop"]["correction_success_rate"] == 1.0

    trace = RevisionTrace.from_dict(report["results"][-1]["revision_trace"])
    buffer = CorrectionBuffer(
        (
            CorrectionRecord(
                record_id="r1",
                prompt=trace.prompt,
                initial_answer=trace.initial_answer,
                revised_answer=trace.revised_answer,
                claims=tuple(item.claim for item in trace.claim_revisions),
                revision_trace=trace.to_dict(),
                verifier_status="verified",
                correction_success=trace.correction_success,
            ),
            CorrectionRecord(
                record_id="r2",
                prompt="unverified",
                initial_answer="wrong",
                revised_answer="right",
                verifier_status="unverified",
                correction_success=True,
            ),
        )
    )
    buffer_path = tmp_path / "buffer.jsonl"
    output_path = tmp_path / "training.jsonl"
    export_report_path = tmp_path / "training-report.json"
    buffer.write_jsonl(buffer_path)

    export_report = export_correction_training_data(
        buffer_path=buffer_path,
        output_jsonl_path=output_path,
        format="sft",
        report_json_path=export_report_path,
        artifact_manifest_path=tmp_path / "training-manifest.json",
    )
    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert export_report["summary"]["exported_record_count"] == 1
    assert rows[0]["metadata"]["record_id"] == "r1"

