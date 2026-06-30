"""Trace-side claim support audits for action receipts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from eigentruth.control.action_audit import ActionAuditSeverity
from eigentruth.control.actions import ActionExecutionStatus, ActionResult
from eigentruth.control.receipts import (
    ActionReceipt,
    action_result_fingerprint,
    json_fingerprint,
    verify_action_receipt,
)
from eigentruth.json_utils import to_jsonable

_CLAIM_REQUEST_ID_KEYS = (
    "action_request_id",
    "action_request_ids",
    "action_result_request_id",
    "action_result_request_ids",
    "receipt_request_id",
    "receipt_request_ids",
    "tool_request_id",
    "tool_request_ids",
    "evidence_action_request_id",
    "evidence_action_request_ids",
    "source_action_request_id",
    "source_action_request_ids",
)
_EVIDENCE_REQUEST_ID_KEYS = (*_CLAIM_REQUEST_ID_KEYS, "request_id", "request_ids")
_RESULT_FINGERPRINT_KEYS = (
    "action_result_fingerprint",
    "action_result_fingerprints",
    "result_fingerprint",
    "result_fingerprints",
    "receipt_result_fingerprint",
    "receipt_result_fingerprints",
)
_OUTPUT_FINGERPRINT_KEYS = (
    "action_output_fingerprint",
    "action_output_fingerprints",
    "output_fingerprint",
    "output_fingerprints",
    "receipt_output_fingerprint",
    "receipt_output_fingerprints",
)


@dataclass(frozen=True)
class ReceiptClaimSupportPolicy:
    """Policy for structural claim-to-receipt audits.

    The audit only checks explicit references. It does not attempt natural
    language entailment between a claim and a tool output.
    """

    accepted_statuses: Sequence[ActionExecutionStatus | str] = (ActionExecutionStatus.SUCCEEDED,)
    require_signed_receipt: bool = False
    warn_on_unsigned_receipt: bool = True

    def __post_init__(self) -> None:
        accepted = tuple(
            status if isinstance(status, ActionExecutionStatus) else ActionExecutionStatus(str(status))
            for status in self.accepted_statuses
        )
        if not accepted:
            raise ValueError("accepted_statuses must be non-empty.")
        object.__setattr__(self, "accepted_statuses", accepted)
        object.__setattr__(self, "require_signed_receipt", _strict_bool(self.require_signed_receipt))
        object.__setattr__(self, "warn_on_unsigned_receipt", _strict_bool(self.warn_on_unsigned_receipt))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready policy payload."""
        return {
            "accepted_statuses": tuple(status.value for status in self.accepted_statuses),
            "require_signed_receipt": self.require_signed_receipt,
            "warn_on_unsigned_receipt": self.warn_on_unsigned_receipt,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ReceiptClaimSupportPolicy":
        """Build a policy from JSON-like data."""
        return cls(
            accepted_statuses=tuple(_as_sequence(data.get("accepted_statuses", ("succeeded",)))),
            require_signed_receipt=_strict_bool(data.get("require_signed_receipt", False)),
            warn_on_unsigned_receipt=_strict_bool(data.get("warn_on_unsigned_receipt", True)),
        )


@dataclass(frozen=True)
class ReceiptClaimReference:
    """One explicit claim/final-answer reference to an action receipt."""

    reference_type: str
    value: str
    source: str
    source_index: int | None = None
    claim_id: str | None = None
    field_path: str | None = None
    text: str | None = None

    def __post_init__(self) -> None:
        reference_type = str(self.reference_type).strip()
        value = str(self.value).strip()
        source = str(self.source).strip()
        if reference_type not in {"request_id", "result_fingerprint", "output_fingerprint"}:
            raise ValueError("reference_type must be request_id, result_fingerprint, or output_fingerprint.")
        if not value:
            raise ValueError("receipt claim reference value must be non-empty.")
        if not source:
            raise ValueError("receipt claim reference source must be non-empty.")
        object.__setattr__(self, "reference_type", reference_type)
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "source_index", None if self.source_index is None else int(self.source_index))
        object.__setattr__(self, "claim_id", None if self.claim_id is None else str(self.claim_id))
        object.__setattr__(self, "field_path", None if self.field_path is None else str(self.field_path))
        object.__setattr__(self, "text", None if self.text is None else str(self.text))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready reference."""
        return {
            "reference_type": self.reference_type,
            "value": self.value,
            "source": self.source,
            "source_index": self.source_index,
            "claim_id": self.claim_id,
            "field_path": self.field_path,
            "text": self.text,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ReceiptClaimReference":
        """Build a reference from JSON-like data."""
        return cls(
            reference_type=str(data["reference_type"]),
            value=str(data["value"]),
            source=str(data["source"]),
            source_index=None if data.get("source_index") is None else int(data["source_index"]),
            claim_id=None if data.get("claim_id") is None else str(data["claim_id"]),
            field_path=None if data.get("field_path") is None else str(data["field_path"]),
            text=None if data.get("text") is None else str(data["text"]),
        )


@dataclass(frozen=True)
class ReceiptClaimSupportIssue:
    """One structural claim-to-receipt support issue."""

    code: str
    severity: ActionAuditSeverity | str
    message: str
    reference: ReceiptClaimReference | Mapping[str, Any] | None = None
    result_index: int | None = None
    request_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        code = str(self.code).strip()
        message = str(self.message).strip()
        if not code:
            raise ValueError("receipt claim support issue code must be non-empty.")
        if not message:
            raise ValueError("receipt claim support issue message must be non-empty.")
        severity = (
            self.severity
            if isinstance(self.severity, ActionAuditSeverity)
            else ActionAuditSeverity(str(self.severity))
        )
        reference = self.reference
        if reference is not None and not isinstance(reference, ReceiptClaimReference):
            reference = ReceiptClaimReference.from_dict(reference)
        object.__setattr__(self, "code", code)
        object.__setattr__(self, "severity", severity)
        object.__setattr__(self, "message", message)
        object.__setattr__(self, "reference", reference)
        object.__setattr__(self, "result_index", None if self.result_index is None else int(self.result_index))
        object.__setattr__(self, "request_id", None if self.request_id is None else str(self.request_id))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready issue."""
        return {
            "code": self.code,
            "severity": self.severity.value,
            "message": self.message,
            "reference": None if self.reference is None else self.reference.to_dict(),
            "result_index": self.result_index,
            "request_id": self.request_id,
            "metadata": to_jsonable(dict(self.metadata)),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ReceiptClaimSupportIssue":
        """Build an issue from JSON-like data."""
        return cls(
            code=str(data["code"]),
            severity=str(data["severity"]),
            message=str(data["message"]),
            reference=data.get("reference"),
            result_index=None if data.get("result_index") is None else int(data["result_index"]),
            request_id=None if data.get("request_id") is None else str(data["request_id"]),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(frozen=True)
class ReceiptClaimSupportReport:
    """JSON-ready structural audit from claims/final answers to action receipts."""

    claim_count: int
    final_answer_evidence_count: int
    action_result_count: int
    references: Sequence[ReceiptClaimReference | Mapping[str, Any]] = ()
    issues: Sequence[ReceiptClaimSupportIssue | Mapping[str, Any]] = ()
    policy: ReceiptClaimSupportPolicy | Mapping[str, Any] = field(default_factory=ReceiptClaimSupportPolicy)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        claim_count = int(self.claim_count)
        evidence_count = int(self.final_answer_evidence_count)
        result_count = int(self.action_result_count)
        if claim_count < 0 or evidence_count < 0 or result_count < 0:
            raise ValueError("report counts must be non-negative.")
        references = tuple(
            ref if isinstance(ref, ReceiptClaimReference) else ReceiptClaimReference.from_dict(ref)
            for ref in self.references
        )
        issues = tuple(
            issue if isinstance(issue, ReceiptClaimSupportIssue) else ReceiptClaimSupportIssue.from_dict(issue)
            for issue in self.issues
        )
        policy = self.policy if isinstance(self.policy, ReceiptClaimSupportPolicy) else (
            ReceiptClaimSupportPolicy.from_dict(self.policy)
        )
        object.__setattr__(self, "claim_count", claim_count)
        object.__setattr__(self, "final_answer_evidence_count", evidence_count)
        object.__setattr__(self, "action_result_count", result_count)
        object.__setattr__(self, "references", references)
        object.__setattr__(self, "issues", issues)
        object.__setattr__(self, "policy", policy)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def passed(self) -> bool:
        """Return whether every explicit reference is supported."""
        return not any(issue.severity is ActionAuditSeverity.ERROR for issue in self.issues)

    def summary(self) -> dict[str, Any]:
        """Return a compact telemetry summary."""
        counts_by_code: dict[str, int] = {}
        counts_by_severity: dict[str, int] = {}
        for issue in self.issues:
            counts_by_code[issue.code] = counts_by_code.get(issue.code, 0) + 1
            severity = issue.severity.value
            counts_by_severity[severity] = counts_by_severity.get(severity, 0) + 1
        referenced_claim_ids = tuple(
            dict.fromkeys(ref.claim_id for ref in self.references if ref.claim_id is not None)
        )
        referenced_evidence_indexes = tuple(
            dict.fromkeys(
                int(ref.source_index)
                for ref in self.references
                if ref.source == "final_answer_evidence" and ref.source_index is not None
            )
        )
        return {
            "available": True,
            "passed": self.passed,
            "claim_count": self.claim_count,
            "final_answer_evidence_count": self.final_answer_evidence_count,
            "action_result_count": self.action_result_count,
            "reference_count": len(self.references),
            "referenced_claim_count": len(referenced_claim_ids),
            "referenced_final_answer_evidence_count": len(referenced_evidence_indexes),
            "issue_count": len(self.issues),
            "error_count": counts_by_severity.get(ActionAuditSeverity.ERROR.value, 0),
            "warning_count": counts_by_severity.get(ActionAuditSeverity.WARNING.value, 0),
            "info_count": counts_by_severity.get(ActionAuditSeverity.INFO.value, 0),
            "counts_by_code": counts_by_code,
            "counts_by_severity": counts_by_severity,
            "unsupported_reference_count": counts_by_severity.get(ActionAuditSeverity.ERROR.value, 0),
            "missing_reference_count": counts_by_code.get("missing_action_result", 0),
            "unreceipted_reference_count": counts_by_code.get("action_result_missing_receipt", 0),
            "failed_result_reference_count": counts_by_code.get("action_result_status_not_accepted", 0),
            "fingerprint_mismatch_reference_count": counts_by_code.get("action_receipt_binding_mismatch", 0),
            "unsigned_reference_count": counts_by_code.get("unsigned_action_receipt", 0),
            "top_issues": tuple(issue.to_dict() for issue in self.issues[:8]),
            "policy": self.policy.to_dict(),
        }

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready report."""
        return {
            "claim_count": self.claim_count,
            "final_answer_evidence_count": self.final_answer_evidence_count,
            "action_result_count": self.action_result_count,
            "references": tuple(ref.to_dict() for ref in self.references),
            "issues": tuple(issue.to_dict() for issue in self.issues),
            "policy": self.policy.to_dict(),
            "metadata": to_jsonable(dict(self.metadata)),
            "summary": self.summary(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ReceiptClaimSupportReport":
        """Build a report from JSON-like data."""
        return cls(
            claim_count=int(data["claim_count"]),
            final_answer_evidence_count=int(data["final_answer_evidence_count"]),
            action_result_count=int(data["action_result_count"]),
            references=tuple(_as_sequence(data.get("references", ()))),
            issues=tuple(_as_sequence(data.get("issues", ()))),
            policy=dict(data.get("policy", {})),
            metadata=dict(data.get("metadata", {})),
        )


def audit_receipt_claim_support(
    trace: Any,
    *,
    policy: ReceiptClaimSupportPolicy | Mapping[str, Any] | None = None,
) -> ReceiptClaimSupportReport:
    """Audit explicit claim/final-answer references against action receipts.

    The audit is intentionally structural: a claim or final-answer evidence item
    must explicitly reference an action request id, result fingerprint, or output
    fingerprint. Referenced action results then need a receipt whose fingerprints
    match the stored result payload. This catches tool-use fabrication and
    traceability gaps without pretending to solve semantic entailment.
    """
    trace_payload = _trace_payload(trace)
    resolved_policy = _coerce_policy(policy)
    action_results = tuple(
        _action_result_payload(item)
        for item in _as_sequence(trace_payload.get("action_results", ()))
    )
    index = _build_result_index(action_results)
    references = _references_from_trace(trace_payload)
    issues: list[ReceiptClaimSupportIssue] = []

    for reference in references:
        matches = _matching_result_records(reference, index)
        if not matches:
            issues.append(ReceiptClaimSupportIssue(
                code="missing_action_result",
                severity=ActionAuditSeverity.ERROR,
                message="Referenced action result was not found in the trace.",
                reference=reference,
            ))
            continue
        supported = False
        pending_warning: ReceiptClaimSupportIssue | None = None
        for record in matches:
            check = _support_issue_for_record(reference, record, resolved_policy)
            if check is None:
                supported = True
                if record["receipt_algorithm"] == "unsigned" and resolved_policy.warn_on_unsigned_receipt:
                    pending_warning = ReceiptClaimSupportIssue(
                        code="unsigned_action_receipt",
                        severity=ActionAuditSeverity.WARNING,
                        message="Referenced action result is receipt-backed but unsigned.",
                        reference=reference,
                        result_index=int(record["index"]),
                        request_id=record["request_id"],
                    )
                break
            issues.append(check)
        if supported and pending_warning is not None:
            issues.append(pending_warning)

    final_answer = _mapping(trace_payload.get("final_answer"))
    return ReceiptClaimSupportReport(
        claim_count=len(_as_sequence(trace_payload.get("claims", ()))),
        final_answer_evidence_count=len(_as_sequence(final_answer.get("evidence", ()))),
        action_result_count=len(action_results),
        references=references,
        issues=tuple(_dedupe_issues(issues)),
        policy=resolved_policy,
        metadata={"source": "eigentruth.control.audit_receipt_claim_support"},
    )


def _support_issue_for_record(
    reference: ReceiptClaimReference,
    record: Mapping[str, Any],
    policy: ReceiptClaimSupportPolicy,
) -> ReceiptClaimSupportIssue | None:
    accepted_statuses = tuple(status.value for status in policy.accepted_statuses)
    if str(record["status"]) not in accepted_statuses:
        return ReceiptClaimSupportIssue(
            code="action_result_status_not_accepted",
            severity=ActionAuditSeverity.ERROR,
            message="Referenced action result status is not accepted by the receipt support policy.",
            reference=reference,
            result_index=int(record["index"]),
            request_id=record["request_id"],
            metadata={"status": record["status"], "accepted_statuses": accepted_statuses},
        )
    if not record["has_receipt"]:
        return ReceiptClaimSupportIssue(
            code="action_result_missing_receipt",
            severity=ActionAuditSeverity.ERROR,
            message="Referenced action result has no receipt.",
            reference=reference,
            result_index=int(record["index"]),
            request_id=record["request_id"],
        )
    if not record["receipt_bound"]:
        return ReceiptClaimSupportIssue(
            code="action_receipt_binding_mismatch",
            severity=ActionAuditSeverity.ERROR,
            message="Referenced action receipt does not match the action result payload.",
            reference=reference,
            result_index=int(record["index"]),
            request_id=record["request_id"],
            metadata={"receipt_issues": tuple(record["receipt_issues"])},
        )
    if policy.require_signed_receipt and record["receipt_algorithm"] != "hmac-sha256":
        return ReceiptClaimSupportIssue(
            code="unsigned_action_receipt",
            severity=ActionAuditSeverity.ERROR,
            message="Referenced action result requires a signed receipt.",
            reference=reference,
            result_index=int(record["index"]),
            request_id=record["request_id"],
        )
    return None


def _build_result_index(action_results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    records = tuple(_result_record(index, result) for index, result in enumerate(action_results))
    by_request_id: dict[str, list[dict[str, Any]]] = {}
    by_result_fingerprint: dict[str, list[dict[str, Any]]] = {}
    by_output_fingerprint: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        request_id = record["request_id"]
        if request_id is not None:
            by_request_id.setdefault(str(request_id), []).append(record)
        by_result_fingerprint.setdefault(str(record["result_fingerprint"]), []).append(record)
        by_output_fingerprint.setdefault(str(record["output_fingerprint"]), []).append(record)
    return {
        "records": records,
        "by_request_id": by_request_id,
        "by_result_fingerprint": by_result_fingerprint,
        "by_output_fingerprint": by_output_fingerprint,
    }


def _result_record(index: int, result: Mapping[str, Any]) -> dict[str, Any]:
    receipts = _receipts_from_result(result)
    bound_receipt: ActionReceipt | None = None
    receipt_issues: tuple[str, ...] = ()
    for receipt in receipts:
        verification = verify_action_receipt(receipt, result=result)
        binding_issues = tuple(issue for issue in verification.issues if issue != "missing_secret")
        if (
            verification.result_fingerprint_match is True
            and verification.output_fingerprint_match is True
            and not binding_issues
            and isinstance(verification.receipt, ActionReceipt)
        ):
            bound_receipt = verification.receipt
            receipt_issues = binding_issues
            break
        receipt_issues = binding_issues or tuple(verification.issues)
    return {
        "index": index,
        "request_id": None if result.get("request_id") is None else str(result.get("request_id")),
        "action": str(result.get("action", "")),
        "status": str(result.get("status", "")),
        "result_fingerprint": action_result_fingerprint(result),
        "output_fingerprint": json_fingerprint(result.get("output", {})),
        "has_receipt": bool(receipts),
        "receipt_bound": bound_receipt is not None,
        "receipt_algorithm": None if bound_receipt is None else bound_receipt.signature_algorithm,
        "receipt_issues": receipt_issues,
    }


def _matching_result_records(
    reference: ReceiptClaimReference,
    index: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    if reference.reference_type == "request_id":
        return tuple(index["by_request_id"].get(reference.value, ()))
    if reference.reference_type == "result_fingerprint":
        return tuple(index["by_result_fingerprint"].get(reference.value.lower(), ()))
    if reference.reference_type == "output_fingerprint":
        return tuple(index["by_output_fingerprint"].get(reference.value.lower(), ()))
    return ()


def _references_from_trace(trace_payload: Mapping[str, Any]) -> tuple[ReceiptClaimReference, ...]:
    references: list[ReceiptClaimReference] = []
    for index, claim in enumerate(_as_sequence(trace_payload.get("claims", ()))):
        claim_payload = _mapping(claim)
        claim_id = None if claim_payload.get("claim_id") is None else str(claim_payload.get("claim_id"))
        text = None if claim_payload.get("text") is None else str(claim_payload.get("text"))
        references.extend(_references_from_mapping(
            claim_payload,
            source="claim",
            source_index=index,
            claim_id=claim_id,
            text=text,
            request_id_keys=_CLAIM_REQUEST_ID_KEYS,
        ))
        metadata = _mapping(claim_payload.get("metadata"))
        references.extend(_references_from_mapping(
            metadata,
            source="claim",
            source_index=index,
            claim_id=claim_id,
            text=text,
            field_prefix="metadata.",
            request_id_keys=_CLAIM_REQUEST_ID_KEYS,
        ))

    final_answer = _mapping(trace_payload.get("final_answer"))
    for index, evidence in enumerate(_as_sequence(final_answer.get("evidence", ()))):
        evidence_payload = _mapping(evidence)
        references.extend(_references_from_mapping(
            evidence_payload,
            source="final_answer_evidence",
            source_index=index,
            text=None if evidence_payload.get("text") is None else str(evidence_payload.get("text")),
            request_id_keys=_EVIDENCE_REQUEST_ID_KEYS,
        ))
        metadata = _mapping(evidence_payload.get("metadata"))
        references.extend(_references_from_mapping(
            metadata,
            source="final_answer_evidence",
            source_index=index,
            field_prefix="metadata.",
            request_id_keys=_EVIDENCE_REQUEST_ID_KEYS,
        ))
    references.extend(_references_from_mapping(
        _mapping(final_answer.get("metadata")),
        source="final_answer_metadata",
        request_id_keys=_EVIDENCE_REQUEST_ID_KEYS,
    ))
    return tuple(dict.fromkeys(references))


def _references_from_mapping(
    payload: Mapping[str, Any],
    *,
    source: str,
    request_id_keys: Sequence[str],
    source_index: int | None = None,
    claim_id: str | None = None,
    text: str | None = None,
    field_prefix: str = "",
) -> tuple[ReceiptClaimReference, ...]:
    references: list[ReceiptClaimReference] = []
    for key in request_id_keys:
        for value in _string_values(payload.get(key)):
            references.append(ReceiptClaimReference(
                reference_type="request_id",
                value=value,
                source=source,
                source_index=source_index,
                claim_id=claim_id,
                field_path=f"{field_prefix}{key}",
                text=text,
            ))
    for key in _RESULT_FINGERPRINT_KEYS:
        for value in _string_values(payload.get(key)):
            references.append(ReceiptClaimReference(
                reference_type="result_fingerprint",
                value=value.lower(),
                source=source,
                source_index=source_index,
                claim_id=claim_id,
                field_path=f"{field_prefix}{key}",
                text=text,
            ))
    for key in _OUTPUT_FINGERPRINT_KEYS:
        for value in _string_values(payload.get(key)):
            references.append(ReceiptClaimReference(
                reference_type="output_fingerprint",
                value=value.lower(),
                source=source,
                source_index=source_index,
                claim_id=claim_id,
                field_path=f"{field_prefix}{key}",
                text=text,
            ))
    return tuple(references)


def _dedupe_issues(issues: Sequence[ReceiptClaimSupportIssue]) -> tuple[ReceiptClaimSupportIssue, ...]:
    deduped = []
    seen = set()
    for issue in issues:
        ref = None if issue.reference is None else (
            issue.reference.reference_type,
            issue.reference.value,
            issue.reference.source,
            issue.reference.source_index,
            issue.reference.claim_id,
        )
        key = (issue.code, issue.severity.value, issue.result_index, issue.request_id, ref)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(issue)
    return tuple(deduped)


def _receipts_from_result(result: Mapping[str, Any]) -> tuple[ActionReceipt | Mapping[str, Any], ...]:
    metadata = _mapping(result.get("metadata"))
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


def _trace_payload(trace: Any) -> dict[str, Any]:
    if isinstance(trace, Mapping):
        return dict(trace)
    to_dict = getattr(trace, "to_dict", None)
    if callable(to_dict):
        payload = to_dict()
        if isinstance(payload, Mapping):
            return dict(payload)
    raise TypeError("trace must be a ProductTrace-like object or JSON mapping.")


def _action_result_payload(result: Any) -> dict[str, Any]:
    if isinstance(result, ActionResult):
        return result.to_dict()
    if not isinstance(result, Mapping):
        raise TypeError("action result must be an ActionResult or JSON object.")
    return {
        "action": str(result.get("action", "")),
        "status": str(result.get("status", "")),
        "output": to_jsonable(result.get("output", {})),
        "metadata": to_jsonable(result.get("metadata", {})),
        "request_id": result.get("request_id"),
        "error": result.get("error"),
    }


def _coerce_policy(
    policy: ReceiptClaimSupportPolicy | Mapping[str, Any] | None,
) -> ReceiptClaimSupportPolicy:
    if policy is None:
        return ReceiptClaimSupportPolicy()
    if isinstance(policy, ReceiptClaimSupportPolicy):
        return policy
    if isinstance(policy, Mapping):
        return ReceiptClaimSupportPolicy.from_dict(policy)
    raise TypeError("policy must be a ReceiptClaimSupportPolicy or JSON object.")


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _as_sequence(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes, bytearray)):
        return (value,)
    if isinstance(value, Sequence):
        return tuple(value)
    return (value,)


def _string_values(value: Any) -> tuple[str, ...]:
    values = []
    for item in _as_sequence(value):
        text = str(item).strip()
        if text:
            values.append(text)
    return tuple(values)


def _strict_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "1", "yes", "on"}:
            return True
        if text in {"false", "0", "no", "off"}:
            return False
    raise ValueError(f"expected a strict bool, got {value!r}")
