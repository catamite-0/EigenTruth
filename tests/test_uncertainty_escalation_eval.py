"""Tests for uncertainty-escalation loop reporting."""

import importlib
import json

import pytest

from eigentruth.eval import uncertainty_escalation_report


def test_uncertainty_escalation_report_summarizes_retrieval_and_quality_delta():
    records = (
        {
            "label": 0,
            "result": _loop_result(
                final_action="accept",
                total_evidence=1,
                final_status="supported",
            ),
        },
        {
            "label": 1,
            "result": _loop_result(
                final_action="retrieve",
                total_evidence=0,
                final_status="insufficient_evidence",
            ),
        },
    )

    report = uncertainty_escalation_report(records)

    assert report["n_total"] == 2
    assert report["uncertainty_escalation"]["triggered_records"] == 2
    assert report["uncertainty_escalation"]["trigger_rate"]["estimate"] == pytest.approx(1.0)
    assert report["uncertainty_escalation"]["verify_claim_total"] == 2
    assert report["uncertainty_escalation"]["retrieval_query_total"] == 2
    assert report["uncertainty_escalation"]["route_counts"] == {"retrieval": 2}
    assert report["uncertainty_escalation"]["uncertainty_reason_counts"] == {"confidence_below:0.65": 2}
    assert report["action_execution"]["retrieval_request_records"] == 2
    assert report["action_execution"]["retrieval_evidence_records"] == 1
    assert report["decision_changes"]["transition_counts"] == {"accept->retrieve": 1}
    assert report["quality"]["initial"]["false_accept_rate"]["estimate"] == pytest.approx(1.0)
    assert report["quality"]["final"]["false_accept_rate"]["estimate"] == pytest.approx(0.0)
    assert report["quality"]["delta"]["accepted_false"] == -1
    assert report["quality"]["delta"]["false_accept_rate"] == pytest.approx(-1.0)
    json.dumps(report, allow_nan=False)


def test_uncertainty_escalation_report_accepts_to_dict_objects_and_explicit_labels():
    class ResultObject:
        def to_dict(self):
            return _loop_result(final_action="accept", total_evidence=1, final_status="supported")

    report = uncertainty_escalation_report((ResultObject(),), labels=("true",))

    assert report["label_summary"]["n_true"] == 1
    assert report["quality"]["final"]["selective_accuracy"]["estimate"] == pytest.approx(1.0)


def test_uncertainty_escalation_report_rejects_partial_embedded_labels():
    records = (
        {"label": 0, "result": _loop_result(final_action="accept", total_evidence=1, final_status="supported")},
        {"result": _loop_result(final_action="retrieve", total_evidence=0, final_status="insufficient_evidence")},
    )

    with pytest.raises(ValueError, match="embedded labels"):
        uncertainty_escalation_report(records)


def test_eval_uncertainty_escalation_cli_reads_jsonl_wrappers(tmp_path):
    module = importlib.import_module("benchmarks.eval_uncertainty_escalation")
    input_path = tmp_path / "loop-results.jsonl"
    output_path = tmp_path / "escalation-report.json"
    rows = (
        {"truth_label": 0, "result": _loop_result(final_action="accept", total_evidence=1, final_status="supported")},
        {
            "truth_label": 1,
            "result": _loop_result(
                final_action="retrieve",
                total_evidence=0,
                final_status="insufficient_evidence",
            ),
        },
    )
    input_path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    exit_code = module.main([
        "--results",
        str(input_path),
        "--label-key",
        "truth_label",
        "--json",
        str(output_path),
    ])

    assert exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["n_total"] == 2
    assert payload["label_summary"]["n_false"] == 1
    assert payload["quality"]["delta"]["accepted_false"] == -1


def _loop_result(*, final_action: str, total_evidence: int, final_status: str) -> dict:
    return {
        "initial_verification_results": (
            {
                "status": "supported",
                "confidence": 0.4,
                "metadata": {"claim_id": "c1"},
            },
        ),
        "initial_decision": {"action": "accept", "risk_level": "low"},
        "action_requests": (
            {"action": "accept", "payload": {}},
            {
                "action": "retrieve",
                "payload": {
                    "retrieval_queries": (
                        {
                            "claim_id": "c1",
                            "query": "Paris is the capital of France.",
                        },
                    ),
                },
            },
        ),
        "action_results": (
            {"action": "accept", "status": "succeeded", "output": {}},
            {
                "action": "retrieve",
                "status": "succeeded",
                "output": {
                    "hits": (
                        {"text": "Paris is the capital of France.", "source": "fixture"},
                    )
                    if total_evidence
                    else (),
                },
            },
        ),
        "retrieval_evidence": {
            "evidence": (
                {"text": "Paris is the capital of France.", "source": "fixture"},
            )
            if total_evidence
            else (),
            "total_evidence": total_evidence,
        },
        "final_verification_results": (
            {
                "status": final_status,
                "confidence": 0.95 if final_status == "supported" else 0.2,
                "metadata": {"claim_id": "c1"},
            },
        ),
        "final_decision": {"action": final_action, "risk_level": "low" if final_action == "accept" else "medium"},
        "uncertainty_escalation_plan": {
            "run_verifier": True,
            "verify_claim_ids": ("c1",),
            "skipped_claim_ids": (),
            "retrieval_queries": (
                {
                    "claim_id": "c1",
                    "query": "Paris is the capital of France.",
                },
            ),
            "route_hints": (
                {
                    "claim_id": "c1",
                    "routes": ("retrieval",),
                },
            ),
            "budget": {
                "uncertainty_escalation": {
                    "uncertainty_reasons": {
                        "c1": ("confidence_below:0.65",),
                    },
                },
            },
        },
    }
