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
    ProductTrace,
    ReceiptActionExecutor,
    action_receipt_summary_from_results,
    attach_action_receipt,
    product_runtime_metrics,
    verify_action_receipt,
)


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
