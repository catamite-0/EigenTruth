"""Claim dependency and coherence helpers."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from eigentruth.verify.protocols import Claim, VerificationResult, VerificationStatus

_DISCOURSE_MARKER_RE = re.compile(
    r"^\s*(?:therefore|thus|hence|consequently|as a result|so)\b|^\s*(?:因此|所以|由此|故而)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ClaimDependency:
    """Directed dependency between two claim ids."""

    parent_id: str
    child_id: str
    relation: str = "requires"
    source: str = "metadata"
    reason: str = ""

    def __post_init__(self) -> None:
        parent_id = str(self.parent_id).strip()
        child_id = str(self.child_id).strip()
        relation = str(self.relation or "requires").strip() or "requires"
        source = str(self.source or "metadata").strip() or "metadata"
        if not parent_id:
            raise ValueError("parent_id must be non-empty")
        if not child_id:
            raise ValueError("child_id must be non-empty")
        if parent_id == child_id:
            raise ValueError("claim dependency cannot point to itself")
        object.__setattr__(self, "parent_id", parent_id)
        object.__setattr__(self, "child_id", child_id)
        object.__setattr__(self, "relation", relation)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "reason", str(self.reason or ""))

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ClaimDependency":
        """Build a dependency from JSON-like data."""
        parent_id = payload.get("parent_id", payload.get("parent", payload.get("claim_id")))
        child_id = payload.get("child_id", payload.get("child"))
        if parent_id is None:
            raise ValueError("dependency parent_id is missing")
        if child_id is None:
            raise ValueError("dependency child_id is missing")
        return cls(
            parent_id=str(parent_id),
            child_id=str(child_id),
            relation=str(payload.get("relation", "requires")),
            source=str(payload.get("source", "metadata")),
            reason=str(payload.get("reason", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "parent_id": self.parent_id,
            "child_id": self.child_id,
            "relation": self.relation,
            "source": self.source,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class ClaimCoherenceIssue:
    """One coherence failure caused by an unsupported dependency."""

    parent_id: str
    child_id: str
    relation: str
    parent_status: str
    child_status: str
    action: str = "blocked_supported_child"
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "parent_id": self.parent_id,
            "child_id": self.child_id,
            "relation": self.relation,
            "parent_status": self.parent_status,
            "child_status": self.child_status,
            "action": self.action,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class ClaimCoherenceReport:
    """Summary of claim dependency checks."""

    dependencies: Sequence[ClaimDependency | Mapping[str, Any]] = ()
    issues: Sequence[ClaimCoherenceIssue | Mapping[str, Any]] = ()
    checked_claim_ids: Sequence[str] = ()
    blocked_claim_ids: Sequence[str] = ()
    missing_parent_ids: Sequence[str] = ()
    total_claims: int = 0

    def __post_init__(self) -> None:
        dependencies = tuple(_dependency_obj(item) for item in self.dependencies)
        issues = tuple(_issue_obj(item) for item in self.issues)
        object.__setattr__(self, "dependencies", dependencies)
        object.__setattr__(self, "issues", issues)
        object.__setattr__(self, "checked_claim_ids", tuple(str(item) for item in self.checked_claim_ids))
        object.__setattr__(self, "blocked_claim_ids", tuple(str(item) for item in self.blocked_claim_ids))
        object.__setattr__(self, "missing_parent_ids", tuple(str(item) for item in self.missing_parent_ids))
        object.__setattr__(self, "total_claims", int(self.total_claims))

    def has_issues(self) -> bool:
        """Return whether any dependency blocked a supported claim."""
        return bool(self.issues)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "total_claims": self.total_claims,
            "checked_claim_ids": tuple(self.checked_claim_ids),
            "dependency_count": len(self.dependencies),
            "dependencies": tuple(item.to_dict() for item in self.dependencies),
            "issue_count": len(self.issues),
            "issues": tuple(item.to_dict() for item in self.issues),
            "blocked_claim_ids": tuple(self.blocked_claim_ids),
            "missing_parent_ids": tuple(self.missing_parent_ids),
        }


def infer_claim_dependencies(
    claims: Sequence[Claim | Mapping[str, Any]],
) -> tuple[ClaimDependency, ...]:
    """Infer dependency edges from claim metadata and discourse markers."""
    dependencies: list[ClaimDependency] = []
    seen: set[tuple[str, str, str]] = set()
    claim_ids = tuple(_claim_id(claim, index) for index, claim in enumerate(claims))
    valid_ids = set(claim_ids)
    for index, claim in enumerate(claims):
        child_id = claim_ids[index]
        metadata = _claim_metadata(claim)
        for item in _metadata_dependency_items(metadata):
            dependency = _dependency_from_metadata_item(item, child_id=child_id, valid_ids=valid_ids)
            if dependency is not None:
                _append_dependency(dependencies, seen, dependency)
        if index > 0 and _DISCOURSE_MARKER_RE.search(_claim_text(claim)):
            _append_dependency(
                dependencies,
                seen,
                ClaimDependency(
                    parent_id=claim_ids[index - 1],
                    child_id=child_id,
                    relation="discourse_marker",
                    source="text_rule",
                    reason="claim starts with a discourse marker",
                ),
            )
    return tuple(dependencies)


def apply_claim_coherence(
    claims: Sequence[Claim | Mapping[str, Any]],
    verification_results: Sequence[VerificationResult | Mapping[str, Any]],
    *,
    dependency_claims: Sequence[Claim | Mapping[str, Any]] | None = None,
    dependencies: Sequence[ClaimDependency | Mapping[str, Any]] | None = None,
) -> tuple[tuple[VerificationResult, ...], ClaimCoherenceReport]:
    """Downgrade supported child claims when required parent claims are not supported."""
    aligned_claims = tuple(claims)
    results = tuple(_verification_result_obj(result) for result in verification_results)
    if len(aligned_claims) != len(results):
        raise ValueError("claims and verification_results must have the same length")

    dependency_source_claims = aligned_claims if dependency_claims is None else tuple(dependency_claims)
    active_dependencies = (
        infer_claim_dependencies(dependency_source_claims)
        if dependencies is None
        else tuple(_dependency_obj(item) for item in dependencies)
    )
    result_claim_ids = tuple(_claim_id(claim, index) for index, claim in enumerate(aligned_claims))
    result_by_id = dict(zip(result_claim_ids, results, strict=True))

    adjusted = list(results)
    issues: list[ClaimCoherenceIssue] = []
    blocked_claim_ids: list[str] = []
    missing_parent_ids: list[str] = []
    result_index_by_id = {claim_id: index for index, claim_id in enumerate(result_claim_ids)}

    issue_keys: set[tuple[str, str, str]] = set()
    while True:
        changed = False
        for dependency in active_dependencies:
            child_index = result_index_by_id.get(dependency.child_id)
            if child_index is None:
                continue
            parent_result = result_by_id.get(dependency.parent_id)
            parent_status = "missing" if parent_result is None else parent_result.status.value
            if parent_result is None and dependency.parent_id not in missing_parent_ids:
                missing_parent_ids.append(dependency.parent_id)
            if parent_result is not None and parent_result.status is VerificationStatus.SUPPORTED:
                continue
            child_result = adjusted[child_index]
            if child_result.status is not VerificationStatus.SUPPORTED:
                continue
            issue = ClaimCoherenceIssue(
                parent_id=dependency.parent_id,
                child_id=dependency.child_id,
                relation=dependency.relation,
                parent_status=parent_status,
                child_status=child_result.status.value,
                reason=_issue_reason(dependency, parent_status),
            )
            issue_key = (issue.parent_id, issue.child_id, issue.relation)
            if issue_key not in issue_keys:
                issue_keys.add(issue_key)
                issues.append(issue)
            if dependency.child_id not in blocked_claim_ids:
                blocked_claim_ids.append(dependency.child_id)
            adjusted[child_index] = _blocked_child_result(child_result, issue)
            result_by_id[dependency.child_id] = adjusted[child_index]
            changed = True
        if not changed:
            break

    report = ClaimCoherenceReport(
        dependencies=active_dependencies,
        issues=tuple(issues),
        checked_claim_ids=result_claim_ids,
        blocked_claim_ids=tuple(blocked_claim_ids),
        missing_parent_ids=tuple(missing_parent_ids),
        total_claims=len(dependency_source_claims),
    )
    return tuple(adjusted), report


def _dependency_obj(item: ClaimDependency | Mapping[str, Any]) -> ClaimDependency:
    if isinstance(item, ClaimDependency):
        return item
    return ClaimDependency.from_mapping(item)


def _issue_obj(item: ClaimCoherenceIssue | Mapping[str, Any]) -> ClaimCoherenceIssue:
    if isinstance(item, ClaimCoherenceIssue):
        return item
    return ClaimCoherenceIssue(
        parent_id=str(item["parent_id"]),
        child_id=str(item["child_id"]),
        relation=str(item.get("relation", "requires")),
        parent_status=str(item.get("parent_status", "missing")),
        child_status=str(item.get("child_status", "supported")),
        action=str(item.get("action", "blocked_supported_child")),
        reason=str(item.get("reason", "")),
    )


def _append_dependency(
    dependencies: list[ClaimDependency],
    seen: set[tuple[str, str, str]],
    dependency: ClaimDependency,
) -> None:
    key = (dependency.parent_id, dependency.child_id, dependency.relation)
    if key in seen:
        return
    seen.add(key)
    dependencies.append(dependency)


def _claim_id(claim: Claim | Mapping[str, Any], index: int) -> str:
    if isinstance(claim, Claim):
        return claim.claim_id or f"c{index + 1}"
    claim_id = claim.get("claim_id")
    return f"c{index + 1}" if claim_id is None else str(claim_id)


def _claim_text(claim: Claim | Mapping[str, Any]) -> str:
    if isinstance(claim, Claim):
        return claim.text
    return str(claim.get("text", ""))


def _claim_metadata(claim: Claim | Mapping[str, Any]) -> Mapping[str, Any]:
    metadata = claim.metadata if isinstance(claim, Claim) else claim.get("metadata", {})
    return metadata if isinstance(metadata, Mapping) else {}


def _metadata_dependency_items(metadata: Mapping[str, Any]) -> tuple[Any, ...]:
    items: list[Any] = []
    for key in ("depends_on", "dependencies", "requires_claims", "requires"):
        if key in metadata:
            items.extend(_as_dependency_items(metadata[key]))
    return tuple(items)


def _as_dependency_items(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, str) or isinstance(value, Mapping):
        return (value,)
    if isinstance(value, Sequence) and not isinstance(value, bytes):
        return tuple(value)
    return (value,)


def _dependency_from_metadata_item(
    item: Any,
    *,
    child_id: str,
    valid_ids: set[str],
) -> ClaimDependency | None:
    if isinstance(item, Mapping):
        parent_id = item.get("parent_id", item.get("parent", item.get("claim_id", item.get("id"))))
        if parent_id is None:
            return None
        return ClaimDependency(
            parent_id=str(parent_id),
            child_id=str(item.get("child_id", child_id)),
            relation=str(item.get("relation", "requires")),
            source=str(item.get("source", "metadata")),
            reason=str(item.get("reason", "")),
        )
    parent_id = str(item).strip()
    if not parent_id or parent_id not in valid_ids:
        return None
    return ClaimDependency(parent_id=parent_id, child_id=child_id)


def _verification_result_obj(result: VerificationResult | Mapping[str, Any]) -> VerificationResult:
    if isinstance(result, VerificationResult):
        return result
    evidence = () if result.get("evidence") is None else result.get("evidence", ())
    if isinstance(evidence, str):
        evidence = (evidence,)
    return VerificationResult(
        status=_coerce_verification_status(result.get("status", VerificationStatus.ERROR.value)),
        confidence=_coerce_confidence(result.get("confidence", 0.0)),
        evidence=tuple(str(item) for item in evidence),
        explanation=str(result.get("explanation", "")),
        metadata=result.get("metadata", {}) if isinstance(result.get("metadata", {}), Mapping) else {},
    )


def _coerce_verification_status(value: Any) -> VerificationStatus:
    if isinstance(value, VerificationStatus):
        return value
    try:
        return VerificationStatus(str(value))
    except ValueError:
        return VerificationStatus.ERROR


def _coerce_confidence(value: Any) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not 0.0 <= confidence <= 1.0:
        return 0.0
    return confidence


def _issue_reason(dependency: ClaimDependency, parent_status: str) -> str:
    detail = f"parent claim {dependency.parent_id} is {parent_status}"
    if dependency.reason:
        return f"{detail}; {dependency.reason}"
    return detail


def _blocked_child_result(
    result: VerificationResult,
    issue: ClaimCoherenceIssue,
) -> VerificationResult:
    metadata = dict(result.metadata)
    metadata["claim_coherence"] = {
        "blocked": True,
        "original_status": result.status.value,
        "parent_id": issue.parent_id,
        "parent_status": issue.parent_status,
        "relation": issue.relation,
    }
    explanation = (
        f"claim coherence blocked supported child: {issue.reason}"
        if result.explanation
        else f"claim coherence blocked supported child because {issue.reason}"
    )
    return VerificationResult(
        status=VerificationStatus.INSUFFICIENT_EVIDENCE,
        confidence=min(result.confidence, 0.5),
        evidence=result.evidence,
        explanation=explanation,
        metadata=metadata,
    )
