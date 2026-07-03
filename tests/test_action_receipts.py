"""Action receipt tests."""

import json

from eigentruth.control import (
    ActionExecutionStatus,
    ActionReceipt,
    ActionReceiptSigner,
    ActionRequest,
    ActionResult,
    ControlAction,
    DryRunActionExecutor,
    FinalAnswer,
    ProductTrace,
    ReceiptActionExecutor,
    RiskLevel,
    action_receipt_summary_from_results,
    attach_action_receipt,
    audit_receipt_claim_support,
    json_fingerprint,
    product_runtime_metrics,
    verify_action_receipt,
)
from eigentruth.verify import Claim


def _request() -> ActionRequest:
    return ActionRequest(
        action=ControlAction.EXECUTE_TOOL,
        reason="reserve inventory",
        payload={"tool": "reserve_inventory", "input": {"sku": "A1", "qty": 2}},
        metadata={"idempotency_key": "reserve:A1:2"},
        request_id="req-1",
    )


def _result() -> ActionResult:
    return ActionResult(
        action=ControlAction.EXECUTE_TOOL,
        status=ActionExecutionStatus.SUCCEEDED,
        output={"reserved": True, "sku": "A1", "qty": 2},
        metadata={"executor": "local_inventory", "side_effects": True},
        request_id="req-1",
    )


def test_action_receipt_signer_verifies_result_and_request_binding():
    signer = ActionReceiptSigner("test-secret", key_id="k1", issuer="unit-test")
    receipt = signer.issue(
        _result(),
        request=_request(),
        created_at="2026-06-30T00:00:00Z",
        metadata={"tool": "reserve_inventory"},
    )

    verification = signer.verify(receipt, result=_result(), request=_request())

    assert receipt.signature_algorithm == "hmac-sha256"
    assert receipt.key_id == "k1"
    assert receipt.issuer == "unit-test"
    assert verification.valid is True
    assert verification.signature_valid is True
    assert verification.result_fingerprint_match is True
    assert verification.request_fingerprint_match is True
    json.dumps(receipt.to_dict())


def test_action_receipt_verification_flags_tampered_output_and_signature():
    signer = ActionReceiptSigner("test-secret")
    receipt = signer.issue(_result(), request=_request())
    tampered_result = ActionResult(
        action=ControlAction.EXECUTE_TOOL,
        status=ActionExecutionStatus.SUCCEEDED,
        output={"reserved": True, "sku": "A1", "qty": 99},
        metadata={"executor": "local_inventory", "side_effects": True},
        request_id="req-1",
    )
    tampered_receipt = ActionReceipt.from_dict({
        **receipt.to_dict(),
        "signature": "f" * 64,
    })

    result_check = signer.verify(receipt, result=tampered_result, request=_request())
    signature_check = verify_action_receipt(tampered_receipt, secret="test-secret")

    assert result_check.valid is False
    assert "result_fingerprint_mismatch" in result_check.issues
    assert "output_fingerprint_mismatch" in result_check.issues
    assert signature_check.valid is False
    assert signature_check.signature_valid is False
    assert "signature_mismatch" in signature_check.issues


def test_receipt_action_executor_attaches_receipt_and_trace_summary():
    signer = ActionReceiptSigner("test-secret")
    request = ActionRequest(
        action=ControlAction.RETRIEVE,
        reason="retrieve evidence",
        payload={"retrieval_targets": ({"claim_id": "c1", "text": "Paris capital France"},)},
        request_id="retrieve-1",
    )
    executor = ReceiptActionExecutor(DryRunActionExecutor(), signer)

    result = executor.execute(request)
    receipt = ActionReceipt.from_dict(result.metadata["action_receipt"])
    verification = signer.verify(receipt, result=result, request=request)
    trace = ProductTrace(actions=(request,), action_results=(result,))
    summary = trace.action_receipt_summary()
    bounded = trace.to_bounded_dict()

    assert verification.valid is True
    assert result.metadata["action_receipt"]["request_id"] == "retrieve-1"
    assert summary["receipt_count"] == 1
    assert summary["signed_receipt_count"] == 1
    assert summary["fingerprint_match_count"] == 1
    assert summary["passed"] is True
    assert bounded["summaries"]["action_receipts"]["passed"] is True
    assert product_runtime_metrics(trace)["action_receipts_passed"] is True
    assert product_runtime_metrics(bounded)["action_receipts_source"] == "bounded_summary"


def test_action_receipt_summary_flags_missing_and_mismatched_receipts():
    signed = attach_action_receipt(_result(), ActionReceiptSigner("test-secret").issue(_result()))
    broken_receipt = {
        **signed.metadata["action_receipt"],
        "result_fingerprint": "0" * 64,
        "signature_algorithm": "unsigned",
        "signature": None,
    }
    broken = ActionResult(
        action=signed.action,
        status=signed.status,
        output=signed.output,
        metadata={"action_receipt": broken_receipt},
        request_id=signed.request_id,
    )
    missing = ActionResult(
        action=ControlAction.ABSTAIN,
        status=ActionExecutionStatus.DRY_RUN,
        output={"message": "no evidence"},
    )

    summary = action_receipt_summary_from_results((broken, missing))

    assert summary["result_count"] == 2
    assert summary["receipt_count"] == 1
    assert summary["missing_receipt_count"] == 1
    assert summary["fingerprint_mismatch_count"] == 1
    assert summary["passed"] is False


def test_receipt_claim_support_audit_links_claims_and_final_answer_evidence():
    signer = ActionReceiptSigner("test-secret")
    request = ActionRequest(
        action=ControlAction.RETRIEVE,
        reason="retrieve supporting evidence",
        payload={"query": "Paris capital France"},
        request_id="retrieve-c1",
    )
    result = ActionResult(
        action=ControlAction.RETRIEVE,
        status=ActionExecutionStatus.SUCCEEDED,
        output={"hits": [{"text": "Paris is the capital of France.", "source": "fixture"}]},
        metadata={"executor": "fixture_retriever"},
        request_id="retrieve-c1",
    )
    receipted = attach_action_receipt(result, signer.issue(result, request=request))
    claim = Claim(
        "Paris is the capital of France.",
        claim_id="c1",
        metadata={
            "action_request_ids": ("retrieve-c1",),
            "receipt_output_fingerprint": json_fingerprint(result.output),
        },
    )
    final_answer = FinalAnswer(
        status="answered",
        text="Paris is the capital of France.",
        answerable=True,
        action=ControlAction.ACCEPT,
        risk_level=RiskLevel.LOW,
        confidence=0.92,
        reason="supported by retrieved evidence",
        evidence=(
            {
                "text": "Paris is the capital of France.",
                "request_id": "retrieve-c1",
            },
        ),
    )
    trace = ProductTrace(
        claims=(claim,),
        actions=(request,),
        action_results=(receipted,),
        final_answer=final_answer,
    )

    report = audit_receipt_claim_support(trace)
    summary = trace.receipt_claim_support_summary()
    bounded = trace.to_bounded_dict()
    metrics = product_runtime_metrics(bounded)

    assert report.passed is True
    assert summary["reference_count"] == 3
    assert summary["referenced_claim_count"] == 1
    assert summary["referenced_final_answer_evidence_count"] == 1
    assert summary["unsupported_reference_count"] == 0
    assert bounded["summaries"]["receipt_claim_support"]["passed"] is True
    assert metrics["receipt_claim_support_source"] == "bounded_summary"
    assert metrics["receipt_claim_support_reference_count"] == 3.0
    assert metrics["receipt_claim_support_passed"] is True


def test_receipt_claim_support_audit_flags_missing_unreceipted_and_failed_results():
    unreceipted = ActionResult(
        action=ControlAction.RETRIEVE,
        status=ActionExecutionStatus.SUCCEEDED,
        output={"hits": []},
        request_id="retrieve-unreceipted",
    )
    failed = ActionResult(
        action=ControlAction.EXECUTE_TOOL,
        status=ActionExecutionStatus.FAILED,
        output={"reserved": False},
        request_id="tool-failed",
        error="inventory unavailable",
    )
    failed_with_receipt = attach_action_receipt(
        failed,
        ActionReceiptSigner("test-secret").issue(failed),
    )
    trace = ProductTrace(
        claims=(
            Claim("Unsupported claim", claim_id="c1", metadata={"action_request_id": "missing"}),
            Claim(
                "Claim with unreceipted evidence",
                claim_id="c2",
                metadata={"action_request_id": "retrieve-unreceipted"},
            ),
            Claim(
                "Claim with failed tool evidence",
                claim_id="c3",
                metadata={"tool_request_id": "tool-failed"},
            ),
        ),
        action_results=(unreceipted, failed_with_receipt),
    )

    summary = trace.receipt_claim_support_summary()
    metrics = product_runtime_metrics(trace)

    assert summary["passed"] is False
    assert summary["reference_count"] == 3
    assert summary["unsupported_reference_count"] == 3
    assert summary["missing_reference_count"] == 1
    assert summary["unreceipted_reference_count"] == 1
    assert summary["failed_result_reference_count"] == 1
    assert summary["counts_by_code"]["missing_action_result"] == 1
    assert summary["counts_by_code"]["action_result_missing_receipt"] == 1
    assert summary["counts_by_code"]["action_result_status_not_accepted"] == 1
    assert metrics["receipt_claim_support_passed"] is False
    assert metrics["receipt_claim_support_unsupported_reference_count"] == 3.0
