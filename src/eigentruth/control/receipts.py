"""Receipt-style verification for executed control actions."""

from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from eigentruth.control.actions import (
    ActionExecutionStatus,
    ActionExecutor,
    ActionRequest,
    ActionResult,
)
from eigentruth.control.policy import ControlAction
from eigentruth.json_utils import strict_json_dumps, to_jsonable


@dataclass(frozen=True)
class ActionReceipt:
    """JSON-ready receipt binding an action result to a signed payload.

    The receipt intentionally binds the action, status, request id, error, and
    output payload. Executor metadata is left outside the bound result
    fingerprint because it often carries local timing, retry, or wrapper fields
    that can change during replay without changing the tool output itself.
    """

    action: ControlAction | str
    status: ActionExecutionStatus | str
    request_id: str | None
    result_fingerprint: str
    output_fingerprint: str
    request_fingerprint: str | None = None
    issuer: str = "eigentruth"
    key_id: str | None = None
    signature: str | None = None
    signature_algorithm: str = "unsigned"
    created_at: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = 1

    def __post_init__(self) -> None:
        action = self.action if isinstance(self.action, ControlAction) else ControlAction(str(self.action))
        status = (
            self.status
            if isinstance(self.status, ActionExecutionStatus)
            else ActionExecutionStatus(str(self.status))
        )
        result_fingerprint = _normalize_fingerprint(self.result_fingerprint, name="result_fingerprint")
        output_fingerprint = _normalize_fingerprint(self.output_fingerprint, name="output_fingerprint")
        request_fingerprint = (
            None
            if self.request_fingerprint is None
            else _normalize_fingerprint(self.request_fingerprint, name="request_fingerprint")
        )
        issuer = str(self.issuer).strip()
        if not issuer:
            raise ValueError("receipt issuer must be non-empty.")
        algorithm = str(self.signature_algorithm).strip().lower()
        if algorithm not in {"unsigned", "hmac-sha256"}:
            raise ValueError("signature_algorithm must be 'unsigned' or 'hmac-sha256'.")
        signature = None if self.signature is None else str(self.signature).strip()
        if algorithm == "unsigned" and signature:
            raise ValueError("unsigned receipts cannot include a signature.")
        if algorithm == "hmac-sha256":
            if not signature:
                raise ValueError("hmac-sha256 receipts require a signature.")
            _normalize_fingerprint(signature, name="signature")
        key_id = None if self.key_id is None else str(self.key_id).strip()
        created_at = None if self.created_at is None else str(self.created_at)
        object.__setattr__(self, "action", action)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "request_id", None if self.request_id is None else str(self.request_id))
        object.__setattr__(self, "result_fingerprint", result_fingerprint)
        object.__setattr__(self, "output_fingerprint", output_fingerprint)
        object.__setattr__(self, "request_fingerprint", request_fingerprint)
        object.__setattr__(self, "issuer", issuer)
        object.__setattr__(self, "key_id", key_id or None)
        object.__setattr__(self, "signature", signature)
        object.__setattr__(self, "signature_algorithm", algorithm)
        object.__setattr__(self, "created_at", created_at)
        object.__setattr__(self, "metadata", dict(self.metadata))
        object.__setattr__(self, "schema_version", int(self.schema_version))

    def unsigned_payload(self) -> dict[str, Any]:
        """Return the canonical payload covered by the receipt signature."""
        return {
            "schema_version": self.schema_version,
            "action": self.action.value,
            "status": self.status.value,
            "request_id": self.request_id,
            "request_fingerprint": self.request_fingerprint,
            "result_fingerprint": self.result_fingerprint,
            "output_fingerprint": self.output_fingerprint,
            "issuer": self.issuer,
            "key_id": self.key_id,
            "signature_algorithm": self.signature_algorithm,
            "created_at": self.created_at,
            "metadata": to_jsonable(dict(self.metadata)),
        }

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable receipt."""
        payload = self.unsigned_payload()
        payload["signature"] = self.signature
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ActionReceipt":
        """Build an action receipt from JSON-like data."""
        return cls(
            action=str(data["action"]),
            status=str(data["status"]),
            request_id=None if data.get("request_id") is None else str(data["request_id"]),
            request_fingerprint=(
                None
                if data.get("request_fingerprint") is None
                else str(data["request_fingerprint"])
            ),
            result_fingerprint=str(data["result_fingerprint"]),
            output_fingerprint=str(data["output_fingerprint"]),
            issuer=str(data.get("issuer", "eigentruth")),
            key_id=None if data.get("key_id") is None else str(data["key_id"]),
            signature=None if data.get("signature") is None else str(data["signature"]),
            signature_algorithm=str(data.get("signature_algorithm", "unsigned")),
            created_at=None if data.get("created_at") is None else str(data["created_at"]),
            metadata=dict(data.get("metadata", {})),
            schema_version=int(data.get("schema_version", 1)),
        )


@dataclass(frozen=True)
class ActionReceiptVerification:
    """Verification result for one action receipt."""

    valid: bool
    issues: Sequence[str] = ()
    signature_valid: bool | None = None
    result_fingerprint_match: bool | None = None
    output_fingerprint_match: bool | None = None
    request_fingerprint_match: bool | None = None
    receipt: ActionReceipt | Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "valid", bool(self.valid))
        object.__setattr__(self, "issues", tuple(str(issue) for issue in self.issues if str(issue).strip()))
        if self.receipt is not None and not isinstance(self.receipt, ActionReceipt):
            object.__setattr__(self, "receipt", ActionReceipt.from_dict(self.receipt))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready verification result."""
        receipt_payload = None
        if isinstance(self.receipt, ActionReceipt):
            receipt_payload = self.receipt.to_dict()
        elif isinstance(self.receipt, Mapping):
            receipt_payload = dict(self.receipt)
        return {
            "valid": self.valid,
            "issues": tuple(self.issues),
            "signature_valid": self.signature_valid,
            "result_fingerprint_match": self.result_fingerprint_match,
            "output_fingerprint_match": self.output_fingerprint_match,
            "request_fingerprint_match": self.request_fingerprint_match,
            "receipt": receipt_payload,
        }


@dataclass(frozen=True)
class ActionReceiptSigner:
    """HMAC-SHA256 signer for local action execution receipts."""

    secret: str | bytes
    key_id: str = "local"
    issuer: str = "eigentruth"

    def __post_init__(self) -> None:
        if not _secret_bytes(self.secret):
            raise ValueError("receipt signer secret must be non-empty.")
        key_id = str(self.key_id).strip()
        issuer = str(self.issuer).strip()
        if not key_id:
            raise ValueError("receipt signer key_id must be non-empty.")
        if not issuer:
            raise ValueError("receipt signer issuer must be non-empty.")
        object.__setattr__(self, "key_id", key_id)
        object.__setattr__(self, "issuer", issuer)

    def issue(
        self,
        result: ActionResult | Mapping[str, Any],
        *,
        request: ActionRequest | Mapping[str, Any] | None = None,
        created_at: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ActionReceipt:
        """Create a signed receipt for an action result."""
        result_payload = _action_result_payload(result)
        request_fingerprint = None if request is None else action_request_fingerprint(request)
        receipt = ActionReceipt(
            action=str(result_payload["action"]),
            status=str(result_payload["status"]),
            request_id=None if result_payload.get("request_id") is None else str(result_payload["request_id"]),
            request_fingerprint=request_fingerprint,
            result_fingerprint=action_result_fingerprint(result_payload),
            output_fingerprint=json_fingerprint(result_payload.get("output", {})),
            issuer=self.issuer,
            key_id=self.key_id,
            signature_algorithm="hmac-sha256",
            created_at=created_at,
            metadata=dict(metadata or {}),
            signature=_PLACEHOLDER_SIGNATURE,
        )
        signature = _hmac_signature(self.secret, receipt.unsigned_payload())
        return ActionReceipt(**{**receipt.to_dict(), "signature": signature})

    def verify(
        self,
        receipt: ActionReceipt | Mapping[str, Any],
        *,
        result: ActionResult | Mapping[str, Any] | None = None,
        request: ActionRequest | Mapping[str, Any] | None = None,
    ) -> ActionReceiptVerification:
        """Verify a receipt signature and optional result/request bindings."""
        return verify_action_receipt(receipt, result=result, request=request, secret=self.secret)


@dataclass(frozen=True)
class ReceiptActionExecutor:
    """Executor wrapper that attaches a signed receipt to each result."""

    executor: ActionExecutor
    signer: ActionReceiptSigner
    receipt_metadata: Mapping[str, Any] = field(default_factory=dict)

    def execute(
        self,
        request: ActionRequest,
        context: Mapping[str, Any] | None = None,
    ) -> ActionResult:
        """Execute one request and attach an action receipt."""
        try:
            result = self.executor.execute(request, context=context)
        except Exception as exc:  # pragma: no cover - defensive adapter boundary
            result = ActionResult(
                action=request.action,
                status=ActionExecutionStatus.FAILED,
                output={},
                metadata={
                    "executor": type(self).__name__,
                    "wrapped_executor": type(self.executor).__name__,
                    "side_effects": None,
                    "side_effect_status": "unknown_after_failure",
                    "possible_side_effects": True,
                    "context": dict(context or {}),
                },
                request_id=request.request_id,
                error=f"wrapped executor failed: {exc}",
            )
        receipt = self.signer.issue(result, request=request, metadata=self.receipt_metadata)
        return attach_action_receipt(result, receipt)

    def execute_many(
        self,
        requests: Sequence[ActionRequest],
        context: Mapping[str, Any] | None = None,
    ) -> tuple[ActionResult, ...]:
        """Execute multiple requests and attach receipts."""
        return tuple(self.execute(request, context=context) for request in requests)


def attach_action_receipt(
    result: ActionResult | Mapping[str, Any],
    receipt: ActionReceipt | Mapping[str, Any],
) -> ActionResult:
    """Return an ``ActionResult`` with ``metadata.action_receipt`` attached."""
    payload = _action_result_payload(result)
    metadata = dict(payload.get("metadata", {}))
    metadata["action_receipt"] = _coerce_receipt(receipt).to_dict()
    return ActionResult(
        action=ControlAction(str(payload["action"])),
        status=ActionExecutionStatus(str(payload["status"])),
        output=dict(payload.get("output", {})),
        metadata=metadata,
        request_id=None if payload.get("request_id") is None else str(payload["request_id"]),
        error=None if payload.get("error") is None else str(payload["error"]),
    )


def verify_action_receipt(
    receipt: ActionReceipt | Mapping[str, Any],
    *,
    result: ActionResult | Mapping[str, Any] | None = None,
    request: ActionRequest | Mapping[str, Any] | None = None,
    secret: str | bytes | None = None,
) -> ActionReceiptVerification:
    """Verify one receipt against an optional result, request, and HMAC secret."""
    try:
        receipt_obj = _coerce_receipt(receipt)
    except Exception as exc:
        return ActionReceiptVerification(valid=False, issues=(f"invalid_receipt:{exc}",))
    issues: list[str] = []
    signature_valid: bool | None = None
    if receipt_obj.signature_algorithm == "hmac-sha256":
        if secret is None:
            issues.append("missing_secret")
            signature_valid = None
        else:
            expected = _hmac_signature(secret, receipt_obj.unsigned_payload())
            signature_valid = hmac.compare_digest(expected, str(receipt_obj.signature))
            if not signature_valid:
                issues.append("signature_mismatch")
    elif receipt_obj.signature_algorithm == "unsigned":
        signature_valid = None

    result_match: bool | None = None
    output_match: bool | None = None
    if result is not None:
        result_payload = _action_result_payload(result)
        result_match = action_result_fingerprint(result_payload) == receipt_obj.result_fingerprint
        output_match = json_fingerprint(result_payload.get("output", {})) == receipt_obj.output_fingerprint
        if not result_match:
            issues.append("result_fingerprint_mismatch")
        if not output_match:
            issues.append("output_fingerprint_mismatch")
        if str(result_payload.get("action")) != receipt_obj.action.value:
            issues.append("action_mismatch")
        if str(result_payload.get("status")) != receipt_obj.status.value:
            issues.append("status_mismatch")

    request_match: bool | None = None
    if request is not None and receipt_obj.request_fingerprint is not None:
        request_match = action_request_fingerprint(request) == receipt_obj.request_fingerprint
        if not request_match:
            issues.append("request_fingerprint_mismatch")

    return ActionReceiptVerification(
        valid=not issues,
        issues=tuple(dict.fromkeys(issues)),
        signature_valid=signature_valid,
        result_fingerprint_match=result_match,
        output_fingerprint_match=output_match,
        request_fingerprint_match=request_match,
        receipt=receipt_obj,
    )


def action_receipt_summary_from_results(
    results: Sequence[ActionResult | Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize receipt coverage and result-fingerprint integrity."""
    result_payloads = tuple(_action_result_payload(result) for result in results)
    receipt_count = 0
    signed_count = 0
    unsigned_count = 0
    invalid_count = 0
    fingerprint_match_count = 0
    fingerprint_mismatch_count = 0
    counts_by_algorithm: dict[str, int] = {}
    issues: list[dict[str, Any]] = []
    for index, result in enumerate(result_payloads):
        receipts = _receipts_from_result_payload(result)
        if not receipts:
            continue
        for receipt in receipts:
            receipt_count += 1
            try:
                receipt_obj = _coerce_receipt(receipt)
            except Exception as exc:
                invalid_count += 1
                issues.append({"result_index": index, "code": "invalid_receipt", "message": str(exc)})
                continue
            algorithm = receipt_obj.signature_algorithm
            counts_by_algorithm[algorithm] = counts_by_algorithm.get(algorithm, 0) + 1
            if algorithm == "hmac-sha256":
                signed_count += 1
            else:
                unsigned_count += 1
            verification = verify_action_receipt(receipt_obj, result=result)
            if verification.result_fingerprint_match is True and verification.output_fingerprint_match is True:
                fingerprint_match_count += 1
            else:
                fingerprint_mismatch_count += 1
                issues.append({
                    "result_index": index,
                    "code": "fingerprint_mismatch",
                    "issues": tuple(verification.issues),
                })
    results_with_receipts = sum(
        1 for result in result_payloads
        if _receipts_from_result_payload(result)
    )
    missing_count = max(len(result_payloads) - results_with_receipts, 0)
    return {
        "available": bool(result_payloads),
        "result_count": len(result_payloads),
        "receipt_count": receipt_count,
        "missing_receipt_count": missing_count,
        "signed_receipt_count": signed_count,
        "unsigned_receipt_count": unsigned_count,
        "invalid_receipt_count": invalid_count,
        "fingerprint_match_count": fingerprint_match_count,
        "fingerprint_mismatch_count": fingerprint_mismatch_count,
        "counts_by_algorithm": counts_by_algorithm,
        "coverage": 0.0 if not result_payloads else receipt_count / len(result_payloads),
        "passed": receipt_count == len(result_payloads) and invalid_count == 0 and fingerprint_mismatch_count == 0,
        "top_issues": tuple(issues[:8]),
    }


def action_result_fingerprint(result: ActionResult | Mapping[str, Any]) -> str:
    """Return the stable receipt fingerprint for an action result."""
    payload = _action_result_payload(result)
    return json_fingerprint({
        "action": payload["action"],
        "status": payload["status"],
        "request_id": payload.get("request_id"),
        "output": payload.get("output", {}),
        "error": payload.get("error"),
    })


def action_request_fingerprint(request: ActionRequest | Mapping[str, Any]) -> str:
    """Return a stable fingerprint for an action request."""
    if isinstance(request, ActionRequest):
        payload = request.to_dict()
    elif isinstance(request, Mapping):
        payload = {
            "action": str(request["action"]),
            "reason": str(request.get("reason", "")),
            "payload": to_jsonable(request.get("payload", {})),
            "metadata": to_jsonable(request.get("metadata", {})),
            "request_id": request.get("request_id"),
        }
    else:
        raise TypeError("action request must be an ActionRequest or JSON object.")
    return json_fingerprint(payload)


def json_fingerprint(payload: Any) -> str:
    """Return a deterministic SHA-256 fingerprint for a JSON-like payload."""
    blob = strict_json_dumps(to_jsonable(payload), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


_PLACEHOLDER_SIGNATURE = "0" * 64


def _coerce_receipt(receipt: ActionReceipt | Mapping[str, Any]) -> ActionReceipt:
    if isinstance(receipt, ActionReceipt):
        return receipt
    if isinstance(receipt, Mapping):
        return ActionReceipt.from_dict(receipt)
    raise TypeError("action receipt must be an ActionReceipt or JSON object.")


def _action_result_payload(result: ActionResult | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(result, ActionResult):
        return result.to_dict()
    if not isinstance(result, Mapping):
        raise TypeError("action result must be an ActionResult or JSON object.")
    return {
        "action": str(result["action"]),
        "status": str(result["status"]),
        "output": to_jsonable(result.get("output", {})),
        "metadata": to_jsonable(result.get("metadata", {})),
        "request_id": result.get("request_id"),
        "error": result.get("error"),
    }


def _receipts_from_result_payload(result: Mapping[str, Any]) -> tuple[ActionReceipt | Mapping[str, Any], ...]:
    metadata = result.get("metadata", {})
    if not isinstance(metadata, Mapping):
        return ()
    raw = metadata.get("action_receipt")
    if raw is None:
        raw = metadata.get("receipt")
    if raw is None:
        return ()
    if isinstance(raw, Mapping) or isinstance(raw, ActionReceipt):
        return (raw,)
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        return tuple(item for item in raw if isinstance(item, Mapping) or isinstance(item, ActionReceipt))
    return ()


def _hmac_signature(secret: str | bytes, payload: Mapping[str, Any]) -> str:
    key = _secret_bytes(secret)
    blob = strict_json_dumps(to_jsonable(payload), sort_keys=True, separators=(",", ":"))
    return hmac.new(key, blob.encode("utf-8"), hashlib.sha256).hexdigest()


def _secret_bytes(secret: str | bytes) -> bytes:
    if isinstance(secret, bytes):
        return secret
    return str(secret).encode("utf-8")


def _normalize_fingerprint(value: str, *, name: str) -> str:
    text = str(value).strip().lower()
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise ValueError(f"{name} must be a 64-character lowercase hex SHA-256 digest.")
    return text
