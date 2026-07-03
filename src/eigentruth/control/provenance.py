"""Trace provenance graph and claim-support audits for ProductTrace payloads."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from eigentruth.control.action_audit import ActionAuditSeverity
from eigentruth.control.policy import ControlAction
from eigentruth.json_utils import to_jsonable


@dataclass(frozen=True)
class TraceProvenanceNode:
    """One typed node in a product trace provenance graph."""

    node_id: str
    node_type: str
    label: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        node_id = str(self.node_id).strip()
        node_type = str(self.node_type).strip()
        if not node_id:
            raise ValueError("trace provenance node_id must be non-empty.")
        if not node_type:
            raise ValueError("trace provenance node_type must be non-empty.")
        object.__setattr__(self, "node_id", node_id)
        object.__setattr__(self, "node_type", node_type)
        object.__setattr__(self, "label", None if self.label is None else str(self.label))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready node payload."""
        return {
            "node_id": self.node_id,
            "node_type": self.node_type,
            "label": self.label,
            "metadata": to_jsonable(dict(self.metadata)),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TraceProvenanceNode":
        """Build a node from a JSON-like mapping."""
        return cls(
            node_id=str(data["node_id"]),
            node_type=str(data["node_type"]),
            label=None if data.get("label") is None else str(data["label"]),
            metadata=dict(_mapping(data.get("metadata"))),
        )


@dataclass(frozen=True)
class TraceProvenanceEdge:
    """One typed relation in a product trace provenance graph."""

    source_id: str
    target_id: str
    relation: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        source_id = str(self.source_id).strip()
        target_id = str(self.target_id).strip()
        relation = str(self.relation).strip()
        if not source_id:
            raise ValueError("trace provenance edge source_id must be non-empty.")
        if not target_id:
            raise ValueError("trace provenance edge target_id must be non-empty.")
        if not relation:
            raise ValueError("trace provenance edge relation must be non-empty.")
        object.__setattr__(self, "source_id", source_id)
        object.__setattr__(self, "target_id", target_id)
        object.__setattr__(self, "relation", relation)
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready edge payload."""
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relation": self.relation,
            "metadata": to_jsonable(dict(self.metadata)),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TraceProvenanceEdge":
        """Build an edge from a JSON-like mapping."""
        return cls(
            source_id=str(data["source_id"]),
            target_id=str(data["target_id"]),
            relation=str(data["relation"]),
            metadata=dict(_mapping(data.get("metadata"))),
        )


@dataclass(frozen=True)
class TraceProvenanceIssue:
    """One structural issue in trace provenance."""

    code: str
    severity: ActionAuditSeverity | str
    message: str
    node_id: str | None = None
    claim_ids: Sequence[str] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        code = str(self.code).strip()
        message = str(self.message).strip()
        if not code:
            raise ValueError("trace provenance issue code must be non-empty.")
        if not message:
            raise ValueError("trace provenance issue message must be non-empty.")
        severity = (
            self.severity
            if isinstance(self.severity, ActionAuditSeverity)
            else ActionAuditSeverity(str(self.severity))
        )
        claim_ids = tuple(str(item).strip() for item in self.claim_ids if str(item).strip())
        object.__setattr__(self, "code", code)
        object.__setattr__(self, "severity", severity)
        object.__setattr__(self, "message", message)
        object.__setattr__(self, "node_id", None if self.node_id is None else str(self.node_id))
        object.__setattr__(self, "claim_ids", claim_ids)
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready issue payload."""
        return {
            "code": self.code,
            "severity": self.severity.value,
            "message": self.message,
            "node_id": self.node_id,
            "claim_ids": tuple(self.claim_ids),
            "metadata": to_jsonable(dict(self.metadata)),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TraceProvenanceIssue":
        """Build an issue from a JSON-like mapping."""
        return cls(
            code=str(data["code"]),
            severity=str(data["severity"]),
            message=str(data["message"]),
            node_id=None if data.get("node_id") is None else str(data["node_id"]),
            claim_ids=tuple(_sequence(data.get("claim_ids", ()))),
            metadata=dict(_mapping(data.get("metadata"))),
        )


@dataclass(frozen=True)
class TraceProvenanceGraph:
    """Typed execution/evidence provenance graph for one trace."""

    trace_id: str | None = None
    nodes: Sequence[TraceProvenanceNode | Mapping[str, Any]] = ()
    edges: Sequence[TraceProvenanceEdge | Mapping[str, Any]] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        trace_id = None if self.trace_id is None else str(self.trace_id).strip() or None
        nodes = tuple(
            node if isinstance(node, TraceProvenanceNode) else TraceProvenanceNode.from_dict(node)
            for node in self.nodes
        )
        edges = tuple(
            edge if isinstance(edge, TraceProvenanceEdge) else TraceProvenanceEdge.from_dict(edge)
            for edge in self.edges
        )
        object.__setattr__(self, "trace_id", trace_id)
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "edges", edges)
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready graph payload."""
        return {
            "trace_id": self.trace_id,
            "nodes": tuple(node.to_dict() for node in self.nodes),
            "edges": tuple(edge.to_dict() for edge in self.edges),
            "metadata": to_jsonable(dict(self.metadata)),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TraceProvenanceGraph":
        """Build a graph from a JSON-like mapping."""
        return cls(
            trace_id=None if data.get("trace_id") is None else str(data["trace_id"]),
            nodes=tuple(_sequence(data.get("nodes", ()))),
            edges=tuple(_sequence(data.get("edges", ()))),
            metadata=dict(_mapping(data.get("metadata"))),
        )


@dataclass(frozen=True)
class TraceProvenanceReport:
    """JSON-ready provenance graph plus structural audit summary."""

    trace_id: str | None = None
    graph: TraceProvenanceGraph | Mapping[str, Any] = field(default_factory=TraceProvenanceGraph)
    issues: Sequence[TraceProvenanceIssue | Mapping[str, Any]] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        trace_id = None if self.trace_id is None else str(self.trace_id).strip() or None
        graph = self.graph if isinstance(self.graph, TraceProvenanceGraph) else (
            TraceProvenanceGraph.from_dict(self.graph)
        )
        issues = tuple(
            issue if isinstance(issue, TraceProvenanceIssue) else TraceProvenanceIssue.from_dict(issue)
            for issue in self.issues
        )
        object.__setattr__(self, "trace_id", trace_id)
        object.__setattr__(self, "graph", graph)
        object.__setattr__(self, "issues", issues)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def passed(self) -> bool:
        """Return whether no error-level provenance issue was found."""
        return not any(issue.severity is ActionAuditSeverity.ERROR for issue in self.issues)

    def summary(self) -> dict[str, Any]:
        """Return compact provenance telemetry."""
        counts_by_node_type: dict[str, int] = {}
        counts_by_relation: dict[str, int] = {}
        counts_by_code: dict[str, int] = {}
        counts_by_severity: dict[str, int] = {}
        for node in self.graph.nodes:
            counts_by_node_type[node.node_type] = counts_by_node_type.get(node.node_type, 0) + 1
        for edge in self.graph.edges:
            counts_by_relation[edge.relation] = counts_by_relation.get(edge.relation, 0) + 1
        for issue in self.issues:
            counts_by_code[issue.code] = counts_by_code.get(issue.code, 0) + 1
            severity = issue.severity.value
            counts_by_severity[severity] = counts_by_severity.get(severity, 0) + 1
        supported_claim_count = int(self.metadata.get("supported_claim_count", 0))
        supported_claim_with_evidence_count = int(
            self.metadata.get("supported_claim_with_evidence_count", 0)
        )
        final_answer_evidence_count = int(self.metadata.get("final_answer_evidence_count", 0))
        final_answer_claim_reference_count = int(
            self.metadata.get("final_answer_claim_reference_count", 0)
        )
        return {
            "available": True,
            "passed": self.passed,
            "trace_id": self.trace_id,
            "node_count": len(self.graph.nodes),
            "edge_count": len(self.graph.edges),
            "claim_count": int(self.metadata.get("claim_count", 0)),
            "supported_claim_count": supported_claim_count,
            "supported_claim_with_evidence_count": supported_claim_with_evidence_count,
            "unsupported_supported_claim_count": max(
                supported_claim_count - supported_claim_with_evidence_count,
                0,
            ),
            "supported_claim_evidence_coverage": _safe_div(
                supported_claim_with_evidence_count,
                supported_claim_count,
            ),
            "action_result_count": int(self.metadata.get("action_result_count", 0)),
            "retrieval_hit_count": counts_by_node_type.get("retrieval_hit", 0),
            "source_count": counts_by_node_type.get("source", 0),
            "final_answer_evidence_count": final_answer_evidence_count,
            "final_answer_claim_reference_count": final_answer_claim_reference_count,
            "final_answer_evidence_reference_rate": _safe_div(
                final_answer_claim_reference_count,
                final_answer_evidence_count,
            ),
            "missing_reference_count": counts_by_code.get("missing_referenced_action_result", 0),
            "issue_count": len(self.issues),
            "error_count": counts_by_severity.get(ActionAuditSeverity.ERROR.value, 0),
            "warning_count": counts_by_severity.get(ActionAuditSeverity.WARNING.value, 0),
            "info_count": counts_by_severity.get(ActionAuditSeverity.INFO.value, 0),
            "counts_by_node_type": counts_by_node_type,
            "counts_by_relation": counts_by_relation,
            "counts_by_code": counts_by_code,
            "counts_by_severity": counts_by_severity,
            "top_issues": tuple(issue.to_dict() for issue in self.issues[:8]),
        }

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready report payload."""
        return {
            "trace_id": self.trace_id,
            "graph": self.graph.to_dict(),
            "issues": tuple(issue.to_dict() for issue in self.issues),
            "metadata": to_jsonable(dict(self.metadata)),
            "summary": self.summary(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TraceProvenanceReport":
        """Build a report from a JSON-like mapping."""
        return cls(
            trace_id=None if data.get("trace_id") is None else str(data["trace_id"]),
            graph=_mapping(data.get("graph")),
            issues=tuple(_sequence(data.get("issues", ()))),
            metadata=dict(_mapping(data.get("metadata"))),
        )


@dataclass(frozen=True)
class EvidenceGraphConsistencyPolicy:
    """Lightweight content checks for trace evidence links.

    The policy is intentionally lexical and dependency-free. It is a provenance
    consistency audit, not a semantic entailment model.
    """

    min_keyword_overlap: float = 0.2
    min_number_recall: float = 1.0
    min_entity_recall: float = 0.5

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "min_keyword_overlap",
            _rate_float(self.min_keyword_overlap, name="min_keyword_overlap"),
        )
        object.__setattr__(
            self,
            "min_number_recall",
            _rate_float(self.min_number_recall, name="min_number_recall"),
        )
        object.__setattr__(
            self,
            "min_entity_recall",
            _rate_float(self.min_entity_recall, name="min_entity_recall"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready policy payload."""
        return {
            "min_keyword_overlap": self.min_keyword_overlap,
            "min_number_recall": self.min_number_recall,
            "min_entity_recall": self.min_entity_recall,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EvidenceGraphConsistencyPolicy":
        """Build a policy from a JSON-like mapping."""
        return cls(
            min_keyword_overlap=float(data.get("min_keyword_overlap", 0.2)),
            min_number_recall=float(data.get("min_number_recall", 1.0)),
            min_entity_recall=float(data.get("min_entity_recall", 0.5)),
        )


@dataclass(frozen=True)
class EvidenceGraphConsistencyRecord:
    """One supported-claim/evidence content-consistency audit row."""

    claim_id: str
    verification_result_index: int
    status: str
    evidence_count: int
    keyword_overlap: float | None = None
    number_recall: float | None = None
    entity_recall: float | None = None
    missing_numbers: Sequence[str] = ()
    missing_entities: Sequence[str] = ()
    claim_keywords: Sequence[str] = ()
    evidence_keywords: Sequence[str] = ()
    claim_numbers: Sequence[str] = ()
    evidence_numbers: Sequence[str] = ()
    claim_entities: Sequence[str] = ()
    evidence_entities: Sequence[str] = ()
    issue_codes: Sequence[str] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        claim_id = str(self.claim_id).strip()
        status = str(self.status).strip()
        if not claim_id:
            raise ValueError("evidence graph consistency claim_id must be non-empty.")
        if not status:
            raise ValueError("evidence graph consistency status must be non-empty.")
        verification_result_index = int(self.verification_result_index)
        evidence_count = int(self.evidence_count)
        if verification_result_index < 0:
            raise ValueError("verification_result_index must be non-negative.")
        if evidence_count < 0:
            raise ValueError("evidence_count must be non-negative.")
        object.__setattr__(self, "claim_id", claim_id)
        object.__setattr__(self, "verification_result_index", verification_result_index)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "evidence_count", evidence_count)
        object.__setattr__(self, "missing_numbers", _string_tuple(self.missing_numbers))
        object.__setattr__(self, "missing_entities", _string_tuple(self.missing_entities))
        object.__setattr__(self, "claim_keywords", _string_tuple(self.claim_keywords))
        object.__setattr__(self, "evidence_keywords", _string_tuple(self.evidence_keywords))
        object.__setattr__(self, "claim_numbers", _string_tuple(self.claim_numbers))
        object.__setattr__(self, "evidence_numbers", _string_tuple(self.evidence_numbers))
        object.__setattr__(self, "claim_entities", _string_tuple(self.claim_entities))
        object.__setattr__(self, "evidence_entities", _string_tuple(self.evidence_entities))
        object.__setattr__(self, "issue_codes", _string_tuple(self.issue_codes))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready consistency record."""
        return {
            "claim_id": self.claim_id,
            "verification_result_index": self.verification_result_index,
            "status": self.status,
            "evidence_count": self.evidence_count,
            "keyword_overlap": self.keyword_overlap,
            "number_recall": self.number_recall,
            "entity_recall": self.entity_recall,
            "missing_numbers": tuple(self.missing_numbers),
            "missing_entities": tuple(self.missing_entities),
            "claim_keywords": tuple(self.claim_keywords),
            "evidence_keywords": tuple(self.evidence_keywords),
            "claim_numbers": tuple(self.claim_numbers),
            "evidence_numbers": tuple(self.evidence_numbers),
            "claim_entities": tuple(self.claim_entities),
            "evidence_entities": tuple(self.evidence_entities),
            "issue_codes": tuple(self.issue_codes),
            "metadata": to_jsonable(dict(self.metadata)),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EvidenceGraphConsistencyRecord":
        """Build a consistency record from a JSON-like mapping."""
        return cls(
            claim_id=str(data["claim_id"]),
            verification_result_index=int(data["verification_result_index"]),
            status=str(data["status"]),
            evidence_count=int(data["evidence_count"]),
            keyword_overlap=_optional_float(data.get("keyword_overlap")),
            number_recall=_optional_float(data.get("number_recall")),
            entity_recall=_optional_float(data.get("entity_recall")),
            missing_numbers=tuple(_sequence(data.get("missing_numbers", ()))),
            missing_entities=tuple(_sequence(data.get("missing_entities", ()))),
            claim_keywords=tuple(_sequence(data.get("claim_keywords", ()))),
            evidence_keywords=tuple(_sequence(data.get("evidence_keywords", ()))),
            claim_numbers=tuple(_sequence(data.get("claim_numbers", ()))),
            evidence_numbers=tuple(_sequence(data.get("evidence_numbers", ()))),
            claim_entities=tuple(_sequence(data.get("claim_entities", ()))),
            evidence_entities=tuple(_sequence(data.get("evidence_entities", ()))),
            issue_codes=tuple(_sequence(data.get("issue_codes", ()))),
            metadata=dict(_mapping(data.get("metadata"))),
        )


@dataclass(frozen=True)
class EvidenceGraphConsistencyReport:
    """JSON-ready evidence graph consistency report for one trace."""

    trace_id: str | None = None
    policy: EvidenceGraphConsistencyPolicy | Mapping[str, Any] = field(
        default_factory=EvidenceGraphConsistencyPolicy
    )
    records: Sequence[EvidenceGraphConsistencyRecord | Mapping[str, Any]] = ()
    issues: Sequence[TraceProvenanceIssue | Mapping[str, Any]] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        trace_id = None if self.trace_id is None else str(self.trace_id).strip() or None
        policy = self.policy if isinstance(self.policy, EvidenceGraphConsistencyPolicy) else (
            EvidenceGraphConsistencyPolicy.from_dict(self.policy)
        )
        records = tuple(
            record
            if isinstance(record, EvidenceGraphConsistencyRecord)
            else EvidenceGraphConsistencyRecord.from_dict(record)
            for record in self.records
        )
        issues = tuple(
            issue if isinstance(issue, TraceProvenanceIssue) else TraceProvenanceIssue.from_dict(issue)
            for issue in self.issues
        )
        object.__setattr__(self, "trace_id", trace_id)
        object.__setattr__(self, "policy", policy)
        object.__setattr__(self, "records", records)
        object.__setattr__(self, "issues", issues)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def passed(self) -> bool:
        """Return whether no error-level consistency issue was found."""
        return not any(issue.severity is ActionAuditSeverity.ERROR for issue in self.issues)

    def summary(self) -> dict[str, Any]:
        """Return compact evidence graph consistency telemetry."""
        counts_by_status: dict[str, int] = {}
        counts_by_code: dict[str, int] = {}
        counts_by_severity: dict[str, int] = {}
        keyword_overlaps = [
            record.keyword_overlap
            for record in self.records
            if record.keyword_overlap is not None
        ]
        number_recalls = [
            record.number_recall for record in self.records if record.number_recall is not None
        ]
        entity_recalls = [
            record.entity_recall for record in self.records if record.entity_recall is not None
        ]
        for record in self.records:
            counts_by_status[record.status] = counts_by_status.get(record.status, 0) + 1
        for issue in self.issues:
            counts_by_code[issue.code] = counts_by_code.get(issue.code, 0) + 1
            severity = issue.severity.value
            counts_by_severity[severity] = counts_by_severity.get(severity, 0) + 1
        supported_claim_count = int(self.metadata.get("supported_claim_count", len(self.records)))
        evaluated_count = sum(
            1 for record in self.records if record.status != "insufficient_evidence"
        )
        consistent_count = counts_by_status.get("consistent", 0)
        inconsistent_count = counts_by_status.get("inconsistent", 0)
        insufficient_count = counts_by_status.get("insufficient_evidence", 0)
        return {
            "available": True,
            "passed": self.passed,
            "trace_id": self.trace_id,
            "policy": self.policy.to_dict(),
            "supported_claim_count": supported_claim_count,
            "record_count": len(self.records),
            "evaluated_supported_claim_count": evaluated_count,
            "consistent_supported_claim_count": consistent_count,
            "inconsistent_supported_claim_count": inconsistent_count,
            "insufficient_evidence_count": insufficient_count,
            "consistency_coverage_rate": _safe_div(evaluated_count, supported_claim_count),
            "supported_claim_consistency_rate": _safe_div(
                consistent_count,
                evaluated_count,
            ),
            "keyword_overlap_mean": _mean(keyword_overlaps),
            "keyword_overlap_min": _minimum(keyword_overlaps),
            "number_recall_mean": _mean(number_recalls),
            "entity_recall_mean": _mean(entity_recalls),
            "low_keyword_overlap_count": counts_by_code.get("low_keyword_overlap", 0),
            "missing_number_count": counts_by_code.get("missing_claim_number", 0),
            "missing_entity_count": counts_by_code.get("missing_claim_entity", 0),
            "cross_claim_retrieval_hit_count": counts_by_code.get(
                "referenced_cross_claim_retrieval_hit",
                0,
            ),
            "issue_count": len(self.issues),
            "error_count": counts_by_severity.get(ActionAuditSeverity.ERROR.value, 0),
            "warning_count": counts_by_severity.get(ActionAuditSeverity.WARNING.value, 0),
            "info_count": counts_by_severity.get(ActionAuditSeverity.INFO.value, 0),
            "counts_by_status": counts_by_status,
            "counts_by_code": counts_by_code,
            "counts_by_severity": counts_by_severity,
            "top_records": tuple(record.to_dict() for record in self.records[:8]),
            "top_issues": tuple(issue.to_dict() for issue in self.issues[:8]),
        }

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready report payload."""
        return {
            "trace_id": self.trace_id,
            "policy": self.policy.to_dict(),
            "records": tuple(record.to_dict() for record in self.records),
            "issues": tuple(issue.to_dict() for issue in self.issues),
            "metadata": to_jsonable(dict(self.metadata)),
            "summary": self.summary(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EvidenceGraphConsistencyReport":
        """Build a report from a JSON-like mapping."""
        return cls(
            trace_id=None if data.get("trace_id") is None else str(data["trace_id"]),
            policy=_mapping(data.get("policy")),
            records=tuple(_sequence(data.get("records", ()))),
            issues=tuple(_sequence(data.get("issues", ()))),
            metadata=dict(_mapping(data.get("metadata"))),
        )


def build_trace_provenance_graph(trace: Any) -> TraceProvenanceGraph:
    """Build a typed provenance graph from a ProductTrace-like payload."""
    payload = _trace_payload(trace)
    builder = _TraceProvenanceBuilder(trace_id=_optional_string(payload.get("request_id")))
    claims = tuple(_mapping(item) for item in _sequence(payload.get("claims", ())) if isinstance(item, Mapping))
    verification_results = tuple(
        _mapping(item) for item in _sequence(payload.get("verification_results", ())) if isinstance(item, Mapping)
    )
    actions = tuple(_mapping(item) for item in _sequence(payload.get("actions", ())) if isinstance(item, Mapping))
    action_results = tuple(
        _mapping(item) for item in _sequence(payload.get("action_results", ())) if isinstance(item, Mapping)
    )
    final_answer = _optional_mapping(payload.get("final_answer"))
    claim_ids = tuple(_claim_node_id(claim, index)[0] for index, claim in enumerate(claims))

    for index, claim in enumerate(claims):
        claim_id, node_id = _claim_node_id(claim, index)
        builder.add_node(
            node_id,
            "claim",
            label=_truncate(str(claim.get("text", "")), 120),
            metadata={
                "claim_id": claim_id,
                "index": index,
                "has_metadata": bool(_mapping(claim.get("metadata"))),
            },
        )

    for index, action in enumerate(actions):
        request_id = _optional_string(action.get("request_id")) or f"action-{index}"
        node_id = f"action_request:{request_id}"
        builder.add_node(
            node_id,
            "action_request",
            label=_action_name(action.get("action")),
            metadata={
                "index": index,
                "request_id": request_id,
                "action": _action_name(action.get("action")),
            },
        )
        for claim_id in _claim_references(action):
            if claim_id in claim_ids:
                builder.add_edge(f"claim:{claim_id}", node_id, "requested_action_for")

    result_nodes_by_request_id: dict[str, str] = {}
    for index, result in enumerate(action_results):
        request_id = _optional_string(result.get("request_id")) or f"action-result-{index}"
        node_id = f"action_result:{request_id}"
        result_nodes_by_request_id[request_id] = node_id
        builder.add_node(
            node_id,
            "action_result",
            label=_action_name(result.get("action")),
            metadata={
                "index": index,
                "request_id": request_id,
                "action": _action_name(result.get("action")),
                "status": str(result.get("status", "")),
                "has_error": result.get("error") is not None,
            },
        )
        if request_id:
            request_node = f"action_request:{request_id}"
            if request_node in builder.nodes:
                builder.add_edge(request_node, node_id, "produced")
        _add_retrieval_hit_nodes(builder, result_node_id=node_id, result=result)

    for index, result in enumerate(verification_results):
        claim_id = _verification_result_claim_id(result, index=index, claim_ids=claim_ids)
        node_id = f"verification_result:{index}"
        status = str(result.get("status", ""))
        builder.add_node(
            node_id,
            "verification_result",
            label=status,
            metadata={
                "index": index,
                "claim_id": claim_id,
                "status": status,
                "confidence": result.get("confidence"),
            },
        )
        if claim_id in claim_ids:
            builder.add_edge(node_id, f"claim:{claim_id}", "verifies")
        for evidence_index, evidence in enumerate(_sequence(result.get("evidence", ()))):
            evidence_node_id = f"evidence:verification:{index}:{evidence_index}"
            builder.add_node(
                evidence_node_id,
                "evidence",
                label=_truncate(str(evidence), 160),
                metadata={
                    "source": "verification_result.evidence",
                    "verification_result_index": index,
                    "claim_id": claim_id,
                },
            )
            builder.add_edge(evidence_node_id, node_id, "supports_verification")
        for request_id in _referenced_request_ids(result):
            target_node_id = result_nodes_by_request_id.get(request_id)
            if target_node_id is not None:
                builder.add_edge(target_node_id, node_id, "evidence_for_verification", {"request_id": request_id})

    if final_answer is not None:
        final_node_id = "final_answer"
        builder.add_node(
            final_node_id,
            "final_answer",
            label=str(final_answer.get("status", "")),
            metadata={
                "status": str(final_answer.get("status", "")),
                "action": _action_name(final_answer.get("action")),
                "answerable": final_answer.get("answerable"),
            },
        )
        for evidence_index, evidence in enumerate(_sequence(final_answer.get("evidence", ()))):
            if not isinstance(evidence, Mapping):
                continue
            evidence_mapping = _mapping(evidence)
            evidence_node_id = f"evidence:final_answer:{evidence_index}"
            builder.add_node(
                evidence_node_id,
                "final_answer_evidence",
                label=_truncate(str(evidence_mapping.get("status", evidence_mapping.get("text", ""))), 160),
                metadata={
                    "index": evidence_index,
                    "claim_id": _optional_string(evidence_mapping.get("claim_id")),
                    "status": str(evidence_mapping.get("status", "")),
                },
            )
            builder.add_edge(evidence_node_id, final_node_id, "supports_final_answer")
            for claim_id in _claim_references(evidence_mapping):
                if claim_id in claim_ids:
                    builder.add_edge(evidence_node_id, f"claim:{claim_id}", "references_claim")
            for request_id in _referenced_request_ids(evidence_mapping):
                target_node_id = result_nodes_by_request_id.get(request_id)
                if target_node_id is not None:
                    builder.add_edge(target_node_id, evidence_node_id, "evidence_for_final_answer", {
                        "request_id": request_id,
                    })

    return builder.to_graph(metadata={
        "audit_version": 1,
        "claim_count": len(claims),
        "verification_result_count": len(verification_results),
        "action_count": len(actions),
        "action_result_count": len(action_results),
        "final_answer_present": final_answer is not None,
    })


def audit_trace_provenance(trace: Any) -> TraceProvenanceReport:
    """Audit trace-level evidence/execution provenance links.

    This is a structural audit. It checks whether supported claims and final
    answer evidence have explicit links to local evidence, action results, or
    source records; it does not perform semantic entailment.
    """
    payload = _trace_payload(trace)
    graph = build_trace_provenance_graph(payload)
    claims = tuple(_mapping(item) for item in _sequence(payload.get("claims", ())) if isinstance(item, Mapping))
    verification_results = tuple(
        _mapping(item) for item in _sequence(payload.get("verification_results", ())) if isinstance(item, Mapping)
    )
    action_results = tuple(
        _mapping(item) for item in _sequence(payload.get("action_results", ())) if isinstance(item, Mapping)
    )
    final_answer = _optional_mapping(payload.get("final_answer"))
    claim_ids = tuple(_claim_node_id(claim, index)[0] for index, claim in enumerate(claims))
    result_by_request_id = {
        request_id: result
        for result in action_results
        if (request_id := _optional_string(result.get("request_id"))) is not None
    }
    issues: list[TraceProvenanceIssue] = []
    supported_claim_count = 0
    supported_claim_with_evidence_count = 0
    referenced_supported_claims: set[str] = set()
    for index, result in enumerate(verification_results):
        status = str(result.get("status", "")).strip()
        if status != "supported":
            continue
        supported_claim_count += 1
        claim_id = _verification_result_claim_id(result, index=index, claim_ids=claim_ids)
        request_ids = _referenced_request_ids(result)
        has_action_reference_evidence = False
        for request_id in request_ids:
            action_result = result_by_request_id.get(request_id)
            if action_result is None:
                issues.append(TraceProvenanceIssue(
                    code="missing_referenced_action_result",
                    severity=ActionAuditSeverity.ERROR,
                    message="supported verification result references an action result that is missing",
                    node_id=f"verification_result:{index}",
                    claim_ids=(claim_id,),
                    metadata={"request_id": request_id, "source": "verification_result"},
                ))
                continue
            result_status = str(action_result.get("status", "")).strip()
            if result_status in {"failed", "timed_out"}:
                issues.append(TraceProvenanceIssue(
                    code="referenced_failed_action_result",
                    severity=ActionAuditSeverity.ERROR,
                    message="supported verification result references a failed action result",
                    node_id=f"verification_result:{index}",
                    claim_ids=(claim_id,),
                    metadata={"request_id": request_id, "status": result_status},
                ))
            else:
                has_action_reference_evidence = True
        has_local_evidence = bool(_sequence(result.get("evidence", ()))) or has_action_reference_evidence
        if has_local_evidence:
            supported_claim_with_evidence_count += 1
        else:
            issues.append(TraceProvenanceIssue(
                code="supported_claim_without_evidence",
                severity=ActionAuditSeverity.WARNING,
                message="supported verification result has no local evidence or action-result reference",
                node_id=f"verification_result:{index}",
                claim_ids=(claim_id,),
                metadata={"source": "verification_result"},
            ))

    final_answer_evidence_count = 0
    final_answer_claim_reference_count = 0
    if final_answer is not None:
        evidence_items = tuple(
            _mapping(item) for item in _sequence(final_answer.get("evidence", ())) if isinstance(item, Mapping)
        )
        final_answer_evidence_count = len(evidence_items)
        for evidence_index, evidence in enumerate(evidence_items):
            evidence_claim_ids = tuple(_claim_references(evidence))
            request_ids = _referenced_request_ids(evidence)
            if evidence_claim_ids:
                final_answer_claim_reference_count += 1
            for claim_id in evidence_claim_ids:
                if claim_id not in claim_ids:
                    issues.append(TraceProvenanceIssue(
                        code="final_answer_evidence_unknown_claim",
                        severity=ActionAuditSeverity.ERROR,
                        message="final answer evidence references a claim id that is not present in the trace",
                        node_id=f"evidence:final_answer:{evidence_index}",
                        claim_ids=(claim_id,),
                        metadata={"claim_id": claim_id},
                    ))
                else:
                    referenced_supported_claims.add(claim_id)
            for request_id in request_ids:
                action_result = result_by_request_id.get(request_id)
                if action_result is None:
                    issues.append(TraceProvenanceIssue(
                        code="missing_referenced_action_result",
                        severity=ActionAuditSeverity.ERROR,
                        message="final answer evidence references an action result that is missing",
                        node_id=f"evidence:final_answer:{evidence_index}",
                        claim_ids=evidence_claim_ids,
                        metadata={"request_id": request_id, "source": "final_answer.evidence"},
                    ))
        if _answered_or_accepted(final_answer) and claims and final_answer_evidence_count == 0:
            issues.append(TraceProvenanceIssue(
                code="answered_without_final_evidence",
                severity=ActionAuditSeverity.WARNING,
                message="answered final output carries claims but no final-answer evidence items",
                node_id="final_answer",
                claim_ids=claim_ids,
                metadata={"claim_count": len(claims)},
            ))
    unsupported_supported_claim_count = max(supported_claim_count - supported_claim_with_evidence_count, 0)
    metadata = {
        "audit_version": 1,
        "claim_count": len(claims),
        "supported_claim_count": supported_claim_count,
        "supported_claim_with_evidence_count": supported_claim_with_evidence_count,
        "unsupported_supported_claim_count": unsupported_supported_claim_count,
        "referenced_supported_claim_count": len(referenced_supported_claims),
        "verification_result_count": len(verification_results),
        "action_result_count": len(action_results),
        "final_answer_present": final_answer is not None,
        "final_answer_evidence_count": final_answer_evidence_count,
        "final_answer_claim_reference_count": final_answer_claim_reference_count,
    }
    return TraceProvenanceReport(
        trace_id=_optional_string(payload.get("request_id")),
        graph=graph,
        issues=tuple(issues),
        metadata=metadata,
    )


def audit_evidence_graph_consistency(
    trace: Any,
    *,
    policy: EvidenceGraphConsistencyPolicy | Mapping[str, Any] | None = None,
) -> EvidenceGraphConsistencyReport:
    """Audit whether supported-claim evidence is content-consistent with claims.

    This check sits on top of trace provenance. It uses simple lexical,
    number, and entity-token coverage over explicitly linked local evidence and
    retrieval hits. It catches stale or cross-wired evidence references without
    claiming semantic entailment.
    """
    payload = _trace_payload(trace)
    policy_obj = (
        EvidenceGraphConsistencyPolicy()
        if policy is None
        else policy
        if isinstance(policy, EvidenceGraphConsistencyPolicy)
        else EvidenceGraphConsistencyPolicy.from_dict(policy)
    )
    claims = tuple(_mapping(item) for item in _sequence(payload.get("claims", ())) if isinstance(item, Mapping))
    verification_results = tuple(
        _mapping(item) for item in _sequence(payload.get("verification_results", ())) if isinstance(item, Mapping)
    )
    action_results = tuple(
        _mapping(item) for item in _sequence(payload.get("action_results", ())) if isinstance(item, Mapping)
    )
    claim_ids = tuple(_claim_node_id(claim, index)[0] for index, claim in enumerate(claims))
    claim_by_id = {
        claim_id: claim
        for index, claim in enumerate(claims)
        for claim_id, _node_id in (_claim_node_id(claim, index),)
    }
    action_result_by_request_id = {
        request_id: result
        for result in action_results
        if (request_id := _optional_string(result.get("request_id"))) is not None
    }
    records: list[EvidenceGraphConsistencyRecord] = []
    issues: list[TraceProvenanceIssue] = []
    supported_claim_count = 0
    for index, result in enumerate(verification_results):
        if str(result.get("status", "")).strip() != "supported":
            continue
        supported_claim_count += 1
        claim_id = _verification_result_claim_id(result, index=index, claim_ids=claim_ids)
        claim = claim_by_id.get(claim_id, {})
        claim_text = _text_from_payload(claim) or _optional_string(result.get("claim")) or ""
        evidence_items = list(_evidence_texts_from_verification_result(result))
        cross_claim_hit_count = 0
        request_ids = _referenced_request_ids(result)
        for request_id in request_ids:
            action_result = action_result_by_request_id.get(request_id)
            if action_result is None:
                continue
            for hit_text, hit_claim_id in _retrieval_hit_texts(action_result):
                if hit_claim_id is not None and hit_claim_id != claim_id:
                    cross_claim_hit_count += 1
                    issues.append(TraceProvenanceIssue(
                        code="referenced_cross_claim_retrieval_hit",
                        severity=ActionAuditSeverity.ERROR,
                        message="supported verification result cites a retrieval hit linked to a different claim",
                        node_id=f"verification_result:{index}",
                        claim_ids=(claim_id,),
                        metadata={
                            "request_id": request_id,
                            "hit_claim_id": hit_claim_id,
                        },
                    ))
                evidence_items.append(hit_text)
        record, record_issues = _consistency_record_for_claim(
            claim_id=claim_id,
            verification_result_index=index,
            claim_text=claim_text,
            evidence_items=tuple(evidence_items),
            policy=policy_obj,
            metadata={
                "request_ids": request_ids,
                "cross_claim_hit_count": cross_claim_hit_count,
            },
        )
        records.append(record)
        issues.extend(record_issues)
    return EvidenceGraphConsistencyReport(
        trace_id=_optional_string(payload.get("request_id")),
        policy=policy_obj,
        records=tuple(records),
        issues=tuple(issues),
        metadata={
            "audit_version": 1,
            "supported_claim_count": supported_claim_count,
            "claim_count": len(claims),
            "verification_result_count": len(verification_results),
        },
    )


def _consistency_record_for_claim(
    *,
    claim_id: str,
    verification_result_index: int,
    claim_text: str,
    evidence_items: Sequence[str],
    policy: EvidenceGraphConsistencyPolicy,
    metadata: Mapping[str, Any],
) -> tuple[EvidenceGraphConsistencyRecord, tuple[TraceProvenanceIssue, ...]]:
    evidence_texts = tuple(text.strip() for text in evidence_items if text.strip())
    claim_features = _TextEvidenceFeatures.from_text(claim_text)
    evidence_features = _TextEvidenceFeatures.from_text("\n".join(evidence_texts))
    issue_codes: list[str] = []
    issues: list[TraceProvenanceIssue] = []
    if not evidence_texts:
        issue_codes.append("no_local_evidence_text")
        issues.append(TraceProvenanceIssue(
            code="no_local_evidence_text",
            severity=ActionAuditSeverity.WARNING,
            message="supported claim has no local evidence text to compare",
            node_id=f"verification_result:{verification_result_index}",
            claim_ids=(claim_id,),
            metadata={"source": "evidence_graph_consistency"},
        ))
        return EvidenceGraphConsistencyRecord(
            claim_id=claim_id,
            verification_result_index=verification_result_index,
            status="insufficient_evidence",
            evidence_count=0,
            claim_keywords=claim_features.keywords,
            claim_numbers=claim_features.numbers,
            claim_entities=claim_features.entities,
            issue_codes=tuple(issue_codes),
            metadata=dict(metadata),
        ), tuple(issues)

    keyword_overlap = _recall(claim_features.keywords, evidence_features.keywords)
    number_recall = _recall(claim_features.numbers, evidence_features.numbers)
    entity_recall = _recall(claim_features.entities, evidence_features.entities)
    missing_numbers = _missing_items(claim_features.numbers, evidence_features.numbers)
    missing_entities = _missing_items(claim_features.entities, evidence_features.entities)

    if keyword_overlap is not None and keyword_overlap < policy.min_keyword_overlap:
        issue_codes.append("low_keyword_overlap")
        issues.append(TraceProvenanceIssue(
            code="low_keyword_overlap",
            severity=ActionAuditSeverity.WARNING,
            message="supported claim evidence has low keyword overlap with the claim",
            node_id=f"verification_result:{verification_result_index}",
            claim_ids=(claim_id,),
            metadata={
                "keyword_overlap": keyword_overlap,
                "min_keyword_overlap": policy.min_keyword_overlap,
            },
        ))
    if number_recall is not None and number_recall < policy.min_number_recall:
        issue_codes.append("missing_claim_number")
        issues.append(TraceProvenanceIssue(
            code="missing_claim_number",
            severity=ActionAuditSeverity.ERROR,
            message="supported claim evidence is missing one or more numeric facts from the claim",
            node_id=f"verification_result:{verification_result_index}",
            claim_ids=(claim_id,),
            metadata={
                "number_recall": number_recall,
                "min_number_recall": policy.min_number_recall,
                "missing_numbers": missing_numbers,
            },
        ))
    if entity_recall is not None and entity_recall < policy.min_entity_recall:
        issue_codes.append("missing_claim_entity")
        issues.append(TraceProvenanceIssue(
            code="missing_claim_entity",
            severity=ActionAuditSeverity.WARNING,
            message="supported claim evidence is missing one or more entity-like tokens from the claim",
            node_id=f"verification_result:{verification_result_index}",
            claim_ids=(claim_id,),
            metadata={
                "entity_recall": entity_recall,
                "min_entity_recall": policy.min_entity_recall,
                "missing_entities": missing_entities,
            },
        ))
    status = "consistent" if not issue_codes else "inconsistent"
    return EvidenceGraphConsistencyRecord(
        claim_id=claim_id,
        verification_result_index=verification_result_index,
        status=status,
        evidence_count=len(evidence_texts),
        keyword_overlap=keyword_overlap,
        number_recall=number_recall,
        entity_recall=entity_recall,
        missing_numbers=missing_numbers,
        missing_entities=missing_entities,
        claim_keywords=claim_features.keywords,
        evidence_keywords=evidence_features.keywords,
        claim_numbers=claim_features.numbers,
        evidence_numbers=evidence_features.numbers,
        claim_entities=claim_features.entities,
        evidence_entities=evidence_features.entities,
        issue_codes=tuple(issue_codes),
        metadata=dict(metadata),
    ), tuple(issues)


class _TraceProvenanceBuilder:
    def __init__(self, *, trace_id: str | None) -> None:
        self.trace_id = trace_id
        self.nodes: dict[str, TraceProvenanceNode] = {}
        self.edges: dict[tuple[str, str, str], TraceProvenanceEdge] = {}

    def add_node(
        self,
        node_id: str,
        node_type: str,
        *,
        label: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if node_id in self.nodes:
            return
        self.nodes[node_id] = TraceProvenanceNode(
            node_id=node_id,
            node_type=node_type,
            label=label,
            metadata=dict(metadata or {}),
        )

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        relation: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if source_id not in self.nodes or target_id not in self.nodes:
            return
        key = (source_id, target_id, relation)
        if key in self.edges:
            return
        self.edges[key] = TraceProvenanceEdge(
            source_id=source_id,
            target_id=target_id,
            relation=relation,
            metadata=dict(metadata or {}),
        )

    def to_graph(self, *, metadata: Mapping[str, Any]) -> TraceProvenanceGraph:
        return TraceProvenanceGraph(
            trace_id=self.trace_id,
            nodes=tuple(self.nodes.values()),
            edges=tuple(self.edges.values()),
            metadata=dict(metadata),
        )


def _add_retrieval_hit_nodes(
    builder: _TraceProvenanceBuilder,
    *,
    result_node_id: str,
    result: Mapping[str, Any],
) -> None:
    output = _mapping(result.get("output"))
    hits: list[Mapping[str, Any]] = []
    for hit in _sequence(output.get("hits", ())):
        if isinstance(hit, Mapping):
            hits.append(_mapping(hit))
    for query_index, query_result in enumerate(_sequence(output.get("hits_by_query", ()))):
        query_mapping = _mapping(query_result)
        for hit in _sequence(query_mapping.get("hits", ())):
            if isinstance(hit, Mapping):
                hit_mapping = dict(_mapping(hit))
                hit_mapping.setdefault("query_index", query_index)
                hits.append(hit_mapping)
    for index, hit in enumerate(hits):
        hit_id = f"{result_node_id}:hit:{index}"
        builder.add_node(
            hit_id,
            "retrieval_hit",
            label=_truncate(str(hit.get("text", hit.get("content", hit.get("title", "")))), 160),
            metadata={
                "index": index,
                "source": _optional_string(hit.get("source")),
                "score": hit.get("score"),
                "claim_id": _optional_string(hit.get("claim_id")),
                "query_index": hit.get("query_index"),
            },
        )
        builder.add_edge(result_node_id, hit_id, "returned")
        source = _optional_string(hit.get("source")) or _optional_string(_mapping(hit.get("metadata")).get("source"))
        if source is not None:
            source_id = f"source:{_stable_fragment(source)}"
            builder.add_node(source_id, "source", label=source, metadata={"source": source})
            builder.add_edge(hit_id, source_id, "from_source")


def _claim_node_id(claim: Mapping[str, Any], index: int) -> tuple[str, str]:
    claim_id = _optional_string(claim.get("claim_id")) or f"claim-{index}"
    return claim_id, f"claim:{claim_id}"


def _claim_references(payload: Mapping[str, Any]) -> tuple[str, ...]:
    claim_ids: list[str] = []
    for key in ("claim_id", "claim_ids", "source_claim_id", "source_claim_ids"):
        value = payload.get(key)
        if key.endswith("_ids") or key == "claim_ids":
            claim_ids.extend(
                claim_id
                for item in _sequence(value)
                if (claim_id := _optional_string(item)) is not None
            )
        else:
            claim_id = _optional_string(value)
            if claim_id is not None:
                claim_ids.append(claim_id)
    metadata = _mapping(payload.get("metadata"))
    if metadata:
        claim_ids.extend(_claim_references(metadata))
    nested_payload = _mapping(payload.get("payload"))
    if nested_payload:
        claim_ids.extend(_claim_references(nested_payload))
    return tuple(dict.fromkeys(claim_ids))


def _verification_result_claim_id(
    result: Mapping[str, Any],
    *,
    index: int,
    claim_ids: Sequence[str],
) -> str:
    metadata = _mapping(result.get("metadata"))
    for value in (
        result.get("claim_id"),
        metadata.get("claim_id"),
        metadata.get("source_claim_id"),
    ):
        claim_id = _optional_string(value)
        if claim_id is not None:
            return claim_id
    result_claim_ids = _sequence(result.get("claim_ids", ())) or _sequence(metadata.get("claim_ids", ()))
    if result_claim_ids:
        claim_id = _optional_string(result_claim_ids[0])
        if claim_id is not None:
            return claim_id
    if index < len(claim_ids):
        return str(claim_ids[index])
    return f"claim-{index}"


def _referenced_request_ids(payload: Mapping[str, Any]) -> tuple[str, ...]:
    values = []
    for value in _request_id_values(payload):
        request_id = _optional_string(value)
        if request_id is not None:
            values.append(request_id)
    return tuple(dict.fromkeys(values))


def _request_id_values(value: Any) -> tuple[Any, ...]:
    if isinstance(value, Mapping):
        values: list[Any] = []
        for key, item in value.items():
            key_text = str(key).lower()
            if key_text in _REQUEST_ID_KEYS or (
                key_text.endswith("_request_id") and "fingerprint" not in key_text
            ):
                values.append(item)
            elif key_text in _REQUEST_ID_SEQUENCE_KEYS:
                values.extend(_sequence(item))
            if key_text in {"metadata", "evidence", "source", "sources", "references", "trace"}:
                values.extend(_request_id_values(item))
        return tuple(values)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        values = []
        for item in value:
            values.extend(_request_id_values(item))
        return tuple(values)
    return ()


def _trace_payload(trace: Any) -> Mapping[str, Any]:
    if hasattr(trace, "to_dict"):
        payload = trace.to_dict()
        if isinstance(payload, Mapping):
            return payload
    if isinstance(trace, Mapping):
        return trace
    raise TypeError("trace must be a ProductTrace-like object or mapping.")


def _answered_or_accepted(final_answer: Mapping[str, Any]) -> bool:
    return (
        str(final_answer.get("status", "")).strip() == "answered"
        or _action_name(final_answer.get("action")) == ControlAction.ACCEPT.value
        or final_answer.get("answerable") is True
    )


def _action_name(value: Any) -> str:
    if isinstance(value, ControlAction):
        return value.value
    return str(value or "").strip()


def _sequence(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(value)
    return ()


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _optional_mapping(value: Any) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: max(limit - 3, 0)] + "..."


def _stable_fragment(text: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in text.strip())[:80]


def _safe_div(numerator: int | float, denominator: int | float) -> float | None:
    if denominator == 0:
        return None
    return float(numerator) / float(denominator)


@dataclass(frozen=True)
class _TextEvidenceFeatures:
    keywords: tuple[str, ...] = ()
    numbers: tuple[str, ...] = ()
    entities: tuple[str, ...] = ()

    @classmethod
    def from_text(cls, text: str) -> "_TextEvidenceFeatures":
        return cls(
            keywords=_keyword_tokens(text),
            numbers=_number_tokens(text),
            entities=_entity_tokens(text),
        )


def _evidence_texts_from_verification_result(result: Mapping[str, Any]) -> tuple[str, ...]:
    evidence_texts: list[str] = []
    for evidence in _sequence(result.get("evidence", ())):
        text = _text_from_payload(evidence)
        if text is not None:
            evidence_texts.append(text)
    metadata = _mapping(result.get("metadata"))
    for key in ("evidence", "evidence_text", "evidence_texts", "source_text", "source_texts"):
        value = metadata.get(key)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            for item in _sequence(value):
                text = _text_from_payload(item)
                if text is not None:
                    evidence_texts.append(text)
        elif key in metadata:
            text = _text_from_payload(metadata.get(key))
            if text is not None:
                evidence_texts.append(text)
    return tuple(dict.fromkeys(evidence_texts))


def _retrieval_hit_texts(action_result: Mapping[str, Any]) -> tuple[tuple[str, str | None], ...]:
    hits: list[tuple[str, str | None]] = []
    output = _mapping(action_result.get("output"))
    for hit in _retrieval_hits_from_output(output):
        text = _text_from_payload(hit)
        if text is None:
            continue
        hit_claim_id = _optional_string(_mapping(hit).get("claim_id"))
        hits.append((text, hit_claim_id))
    return tuple(hits)


def _retrieval_hits_from_output(output: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    hits: list[Mapping[str, Any]] = []
    for hit in _sequence(output.get("hits", ())):
        if isinstance(hit, Mapping):
            hits.append(_mapping(hit))
    for query_result in _sequence(output.get("hits_by_query", ())):
        query_mapping = _mapping(query_result)
        for hit in _sequence(query_mapping.get("hits", ())):
            if isinstance(hit, Mapping):
                hits.append(_mapping(hit))
    return tuple(hits)


def _text_from_payload(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value.strip() or None
    if isinstance(value, Mapping):
        for key in ("text", "content", "snippet", "title", "source_text", "claim_text"):
            text = _optional_string(value.get(key))
            if text is not None:
                return text
    return _optional_string(value)


def _keyword_tokens(text: str) -> tuple[str, ...]:
    tokens: list[str] = []
    for match in _WORD_RE.finditer(text):
        token = match.group(0).lower().strip("_-'")
        if len(token) < 3 or token in _STOPWORDS or _NUMBER_RE.fullmatch(token):
            continue
        tokens.append(token)
    return tuple(dict.fromkeys(tokens))


def _number_tokens(text: str) -> tuple[str, ...]:
    tokens: list[str] = []
    for match in _NUMBER_RE.finditer(text):
        token = match.group(0).replace(",", "").strip()
        if token:
            tokens.append(token)
    return tuple(dict.fromkeys(tokens))


def _entity_tokens(text: str) -> tuple[str, ...]:
    entities: list[str] = []
    for match in _WORD_RE.finditer(text):
        raw = match.group(0).strip("_-'")
        if len(raw) < 2:
            continue
        lowered = raw.lower()
        if lowered in _STOPWORDS or _NUMBER_RE.fullmatch(raw):
            continue
        if raw[0].isupper() or raw.isupper():
            entities.append(lowered)
    return tuple(dict.fromkeys(entities))


def _recall(reference: Sequence[str], observed: Sequence[str]) -> float | None:
    reference_set = set(reference)
    if not reference_set:
        return None
    observed_set = set(observed)
    return len(reference_set & observed_set) / len(reference_set)


def _missing_items(reference: Sequence[str], observed: Sequence[str]) -> tuple[str, ...]:
    observed_set = set(observed)
    return tuple(item for item in reference if item not in observed_set)


def _mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return sum(float(value) for value in values) / len(values)


def _minimum(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return min(float(value) for value in values)


def _rate_float(value: Any, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1].")
    return result


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _string_tuple(values: Sequence[Any]) -> tuple[str, ...]:
    return tuple(str(value) for value in values)


_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_'’-]*")
_NUMBER_RE = re.compile(r"(?<![A-Za-z0-9_])-?\d+(?:,\d{3})*(?:\.\d+)?%?")
_STOPWORDS = {
    "about",
    "after",
    "again",
    "against",
    "also",
    "and",
    "are",
    "because",
    "been",
    "before",
    "being",
    "between",
    "both",
    "but",
    "can",
    "did",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "her",
    "his",
    "into",
    "its",
    "more",
    "not",
    "off",
    "onto",
    "our",
    "out",
    "over",
    "she",
    "that",
    "the",
    "their",
    "them",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "was",
    "were",
    "when",
    "where",
    "which",
    "while",
    "who",
    "with",
    "you",
    "your",
}


_REQUEST_ID_KEYS = {
    "request_id",
    "action_request_id",
    "action_result_request_id",
    "receipt_request_id",
    "tool_request_id",
    "evidence_action_request_id",
    "source_action_request_id",
    "retrieval_request_id",
    "source_request_id",
}

_REQUEST_ID_SEQUENCE_KEYS = {
    "request_ids",
    "action_request_ids",
    "action_result_request_ids",
    "receipt_request_ids",
    "tool_request_ids",
    "evidence_action_request_ids",
    "source_action_request_ids",
    "retrieval_request_ids",
    "source_request_ids",
}
