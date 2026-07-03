"""Audit world-model rule input tasks before deterministic execution.

Rule-input plans are non-evidence worklists. Before collecting values or
executing candidate rules, this audit checks whether the requested rule family
matches the question intent closely enough to be worth filling. It does not
execute rules, does not use labels, and does not recover model answers from
upstream queues. Mismatched rows are emitted as requeue suggestions for a later
adapter pass.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eigentruth.json_utils import strict_json_dumps  # noqa: E402
from eigentruth.registry import ArtifactRegistry, build_artifact_manifest  # noqa: E402

WORKFLOW = "world_model_rule_input_plan_audit"
SOURCE_WORKFLOW = "world_model_rule_input_collection_plan"
RESERVED_FIELDS = {"answer", "answers", "label", "labels", "model_answer", "record_index", "score_label"}
COLLECTION_FAMILY_BY_RULE_FAMILY = {
    "quantity_or_arithmetic": "numeric_rule_input_collection",
    "entity_disambiguation": "entity_role_rule_input_collection",
    "temporal_consistency": "temporal_snapshot_rule_input_collection",
    "causal_or_procedural": "mechanism_rule_input_collection",
}


def audit_world_model_rule_input_plan(
    *,
    input_tasks: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a JSON-ready audit report for rule-input tasks."""
    audited = tuple(_audit_task(task, ordinal=idx) for idx, task in enumerate(input_tasks, start=1))
    requeue = tuple(_requeue_suggestion(row) for row in audited if row["recommended_rule_family"])
    summary = _summary(audited=audited, requeue=requeue, source_count=len(input_tasks))
    return {
        "schema_version": 1,
        "workflow": WORKFLOW,
        "status": _status(summary),
        "scope": (
            "Pre-execution audit for deterministic world-model rule input tasks. "
            "Findings are routing and input-contract checks only; they are not "
            "verifier evidence and do not promote any rule result."
        ),
        "source": {
            "input_task_workflow": SOURCE_WORKFLOW,
            "input_task_count": len(input_tasks),
        },
        "label_usage": {
            "labels_used_for_audit": False,
            "labels_copied_to_audit": False,
            "model_answers_used_for_audit": False,
            "model_answers_copied_to_audit": False,
            "audit_rows_are_verifier_evidence": False,
            "requeue_suggestions_are_verifier_evidence": False,
        },
        "summary": summary,
        "audited_tasks": audited,
        "requeue_suggestions": requeue,
        "metadata": dict(metadata or {}),
    }


def run(
    *,
    input_tasks_path: str | Path,
    output_dir: str | Path,
    report_json_path: str | Path | None = None,
    audited_tasks_path: str | Path | None = None,
    requeue_suggestions_path: str | Path | None = None,
    artifact_manifest_path: str | Path | None = None,
    registry_path: str | Path | None = None,
    name: str | None = None,
    version: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    compact_json: bool = False,
) -> dict[str, Any]:
    """Audit, write, manifest, and optionally register a rule-input plan."""
    if registry_path is not None and (not name or not version):
        raise ValueError("registry_path requires name and version.")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report_path = Path(report_json_path or output / "world-model-rule-input-plan-audit.json")
    audited_path = Path(audited_tasks_path or output / "audited-rule-input-tasks.jsonl")
    requeue_path = Path(requeue_suggestions_path or output / "rule-input-requeue-suggestions.jsonl")
    manifest_path = Path(artifact_manifest_path or output / "artifact-manifest.json")

    input_tasks = _load_jsonl_mappings(input_tasks_path)
    payload = audit_world_model_rule_input_plan(
        input_tasks=input_tasks,
        metadata=metadata,
    )
    payload = dict(payload)
    payload["paths"] = {
        "input_tasks": str(input_tasks_path),
        "report": str(report_path),
        "audited_tasks": str(audited_path),
        "requeue_suggestions": str(requeue_path),
        "artifact_manifest": str(manifest_path),
    }
    _write_json(report_path, payload, compact=compact_json)
    _write_jsonl(audited_path, payload["audited_tasks"], compact=compact_json)
    _write_jsonl(requeue_path, payload["requeue_suggestions"], compact=compact_json)
    manifest = build_artifact_manifest(
        {
            "world_model_rule_input_plan_audit": report_path,
            "audited_rule_input_tasks": audited_path,
            "rule_input_requeue_suggestions": requeue_path,
            "rule_input_tasks": Path(input_tasks_path),
        },
        root=manifest_path.parent,
        metadata={
            "workflow": WORKFLOW,
            "status": payload["status"],
            "task_count": payload["summary"]["task_count"],
            "finding_count": payload["summary"]["finding_count"],
            "requeue_suggestion_count": payload["summary"]["requeue_suggestion_count"],
            **dict(metadata or {}),
        },
    )
    _write_json(manifest_path, manifest, compact=compact_json)

    if registry_path is not None:
        assert name is not None and version is not None
        ArtifactRegistry.load_json(registry_path).record_report(
            name=name,
            version=version,
            path=report_path,
            metadata={
                "workflow": WORKFLOW,
                "status": payload["status"],
                "task_count": payload["summary"]["task_count"],
                "finding_count": payload["summary"]["finding_count"],
                "requeue_suggestion_count": payload["summary"]["requeue_suggestion_count"],
                "artifact_manifest": str(manifest_path),
                **dict(metadata or {}),
            },
        ).save_json()
    return payload


def _audit_task(task: Mapping[str, Any], *, ordinal: int) -> dict[str, Any]:
    sanitized = _sanitize_task(task)
    profile = _question_profile(
        question=str(sanitized.get("question") or ""),
        question_type=str(sanitized.get("question_type") or ""),
    )
    findings = _findings(sanitized, profile=profile)
    recommended_family = _recommended_rule_family(sanitized, profile=profile, findings=findings)
    return {
        "schema_version": 1,
        "workflow": WORKFLOW,
        "task_id": str(sanitized.get("task_id") or f"rule-input-task-{ordinal:04d}"),
        "source_request_id": str(sanitized.get("source_request_id") or ""),
        "target_id": str(sanitized.get("target_id") or ""),
        "rule_family": str(sanitized.get("rule_family") or ""),
        "collection_family": str(sanitized.get("collection_family") or ""),
        "priority": str(sanitized.get("priority") or ""),
        "question": str(sanitized.get("question") or ""),
        "question_type": str(sanitized.get("question_type") or ""),
        "question_profile": profile,
        "findings": findings,
        "finding_count": len(findings),
        "recommended_rule_family": recommended_family,
        "recommended_collection_family": (
            COLLECTION_FAMILY_BY_RULE_FAMILY.get(recommended_family, "world_model_rule_input_collection")
            if recommended_family
            else ""
        ),
        "recommended_action": _recommended_action(findings=findings, recommended_family=recommended_family),
        "not_verifier_evidence": True,
    }


def _findings(task: Mapping[str, Any], *, profile: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    family = str(task.get("rule_family") or "")
    findings: list[dict[str, Any]] = []
    if task.get("not_verifier_evidence") is not True:
        findings.append(_finding(
            code="rule_input_task_not_marked_non_evidence",
            severity="blocker",
            detail="rule-input tasks must remain non-evidence before deterministic execution",
        ))
    if family == "quantity_or_arithmetic":
        if profile["entity_or_role_intent"] and not profile["direct_numeric_intent"]:
            findings.append(_finding(
                code="quantity_rule_for_entity_or_role_question",
                severity="high",
                detail="question asks for a person, role, or entity rather than a numeric value",
            ))
        elif not profile["direct_numeric_intent"]:
            findings.append(_finding(
                code="quantity_rule_without_direct_numeric_intent",
                severity="medium",
                detail="numeric rule family is weakly supported by the question wording",
            ))
        if not _candidate_claim_binding_available(task):
            findings.append(_finding(
                code="numeric_rule_missing_candidate_claim_binding",
                severity="medium",
                detail=(
                    "numeric execution needs an explicit candidate claim value or calculation target; "
                    "the audit will not recover model answers from upstream queue rows"
                ),
            ))
    elif family == "temporal_consistency":
        if not profile["temporal_intent"]:
            findings.append(_finding(
                code="temporal_rule_without_temporal_intent",
                severity="medium",
                detail="temporal rule family is weakly supported by the question wording",
            ))
    elif family == "entity_disambiguation":
        if profile["direct_numeric_intent"] and not profile["entity_or_role_intent"]:
            findings.append(_finding(
                code="entity_rule_for_numeric_question",
                severity="medium",
                detail="entity disambiguation may be weaker than a numeric rule for this question",
            ))
    elif family and family not in COLLECTION_FAMILY_BY_RULE_FAMILY:
        findings.append(_finding(
            code="unknown_rule_family",
            severity="medium",
            detail=f"rule family {family!r} has no specialized collection contract",
        ))
    return tuple(findings)


def _recommended_rule_family(
    task: Mapping[str, Any],
    *,
    profile: Mapping[str, Any],
    findings: Sequence[Mapping[str, Any]],
) -> str:
    codes = {str(item.get("code") or "") for item in findings}
    current = str(task.get("rule_family") or "")
    if "rule_input_task_not_marked_non_evidence" in codes:
        return ""
    if "quantity_rule_for_entity_or_role_question" in codes:
        return "entity_disambiguation"
    if current == "quantity_or_arithmetic" and profile["temporal_intent"] and not profile["direct_numeric_intent"]:
        return "temporal_consistency"
    if current == "temporal_consistency" and profile["entity_or_role_intent"] and not profile["temporal_intent"]:
        return "entity_disambiguation"
    return ""


def _requeue_suggestion(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "workflow": WORKFLOW,
        "source_request_id": str(row.get("source_request_id") or ""),
        "target_id": str(row.get("target_id") or ""),
        "task_id": str(row.get("task_id") or ""),
        "current_rule_family": str(row.get("rule_family") or ""),
        "recommended_rule_family": str(row.get("recommended_rule_family") or ""),
        "recommended_collection_family": str(row.get("recommended_collection_family") or ""),
        "recommended_action": "requeue_rule_input_task",
        "question": str(row.get("question") or ""),
        "question_type": str(row.get("question_type") or ""),
        "reason_codes": tuple(str(item.get("code") or "") for item in _mapping_sequence(row.get("findings"))),
        "not_verifier_evidence": True,
    }


def _question_profile(*, question: str, question_type: str) -> dict[str, Any]:
    text = question.strip()
    lower = text.casefold()
    normalized = re.sub(r"\s+", " ", lower)
    direct_numeric = _has_direct_numeric_intent(normalized, question_type=question_type)
    entity_or_role = _has_entity_or_role_intent(normalized, question_type=question_type)
    temporal = _has_temporal_intent(normalized, question_type=question_type)
    return {
        "direct_numeric_intent": direct_numeric,
        "entity_or_role_intent": entity_or_role,
        "temporal_intent": temporal,
        "has_digits": bool(re.search(r"\d", normalized)),
        "starts_with": _first_word(normalized),
    }


def _has_direct_numeric_intent(text: str, *, question_type: str) -> bool:
    if question_type == "quantity":
        return True
    numeric_phrases = (
        "how many",
        "how much",
        "what percent",
        "what percentage",
        "what proportion",
        "population",
        "number of",
        "amount of",
        "rate of",
        "ratio",
        "price",
        "cost",
    )
    return any(phrase in text for phrase in numeric_phrases)


def _has_entity_or_role_intent(text: str, *, question_type: str) -> bool:
    if question_type == "person":
        return True
    entity_phrases = (
        "who ",
        "whose ",
        "his name",
        "her name",
        "their name",
        "named ",
        "founder",
        "producer",
        "president",
        "physically travel",
        "will you see",
    )
    return text.startswith(("who ", "whose ")) or any(phrase in text for phrase in entity_phrases)


def _has_temporal_intent(text: str, *, question_type: str) -> bool:
    if question_type == "temporal":
        return True
    temporal_phrases = (
        "recent",
        "decade",
        "as of",
        "currently",
        "current",
        "today",
        "now",
        "happened to",
        "over time",
    )
    return any(phrase in text for phrase in temporal_phrases)


def _candidate_claim_binding_available(task: Mapping[str, Any]) -> bool:
    metadata = _mapping(task.get("metadata"))
    return any(
        bool(_clean(task.get(key)) or _clean(metadata.get(key)))
        for key in ("candidate_claim_value", "candidate_claim", "claim_value", "calculation.expected")
    )


def _recommended_action(*, findings: Sequence[Mapping[str, Any]], recommended_family: str) -> str:
    if recommended_family:
        return "requeue_rule_input_task"
    if any(str(item.get("severity")) == "blocker" for item in findings):
        return "block_until_task_boundary_is_rebuilt"
    if findings:
        return "collect_missing_inputs_with_review"
    return "collect_missing_inputs"


def _finding(*, code: str, severity: str, detail: str) -> dict[str, str]:
    return {"code": code, "severity": severity, "detail": detail}


def _summary(
    *,
    audited: Sequence[Mapping[str, Any]],
    requeue: Sequence[Mapping[str, Any]],
    source_count: int,
) -> dict[str, Any]:
    finding_counts: Counter[str] = Counter()
    severity_counts: Counter[str] = Counter()
    family_counts = Counter(str(row.get("rule_family") or "") for row in audited)
    recommended_family_counts = Counter(str(row.get("recommended_rule_family") or "") for row in audited)
    action_counts = Counter(str(row.get("recommended_action") or "") for row in audited)
    for row in audited:
        for finding in _mapping_sequence(row.get("findings")):
            finding_counts[str(finding.get("code") or "")] += 1
            severity_counts[str(finding.get("severity") or "")] += 1
    return {
        "source_task_count": int(source_count),
        "task_count": len(audited),
        "finding_count": sum(finding_counts.values()),
        "task_with_findings_count": sum(1 for row in audited if int(row.get("finding_count") or 0) > 0),
        "requeue_suggestion_count": len(requeue),
        "pass_count": sum(1 for row in audited if int(row.get("finding_count") or 0) == 0),
        "rule_family_counts": _sorted_counter(family_counts),
        "recommended_rule_family_counts": _sorted_counter(recommended_family_counts),
        "finding_counts": _sorted_counter(finding_counts),
        "severity_counts": _sorted_counter(severity_counts),
        "recommended_action_counts": _sorted_counter(action_counts),
    }


def _status(summary: Mapping[str, Any]) -> str:
    if int(summary.get("task_count", 0)) == 0:
        return "empty"
    if int(summary.get("requeue_suggestion_count", 0)) > 0:
        return "needs_requeue"
    if int(summary.get("finding_count", 0)) > 0:
        return "needs_review"
    return "pass"


def _sanitize_task(task: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): value for key, value in task.items() if str(key) not in RESERVED_FIELDS}


def _load_jsonl_mappings(path: str | Path) -> tuple[Mapping[str, Any], ...]:
    rows = []
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, Mapping):
            raise ValueError(f"{path}:{line_no} must contain a JSON object.")
        rows.append(_sanitize_task(dict(row)))
    return tuple(rows)


def _write_json(path: str | Path, payload: Mapping[str, Any], *, compact: bool) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if compact:
        text = strict_json_dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    else:
        text = strict_json_dumps(payload, indent=2, sort_keys=True) + "\n"
    output.write_text(text, encoding="utf-8")


def _write_jsonl(path: str | Path, rows: Sequence[Mapping[str, Any]], *, compact: bool) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if compact:
        text = "".join(strict_json_dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows)
    else:
        text = "".join(strict_json_dumps(row, sort_keys=True) + "\n" for row in rows)
    output.write_text(text, encoding="utf-8")


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _mapping_sequence(value: Any) -> tuple[Mapping[str, Any], ...]:
    if isinstance(value, Mapping):
        return (value,)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(item for item in value if isinstance(item, Mapping))
    return ()


def _first_word(value: str) -> str:
    match = re.search(r"\w+", value)
    return "" if match is None else match.group(0)


def _clean(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _sorted_counter(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted(((key, value) for key, value in counter.items() if key), key=lambda item: (-item[1], item[0])))


def _parse_metadata(values: Sequence[str]) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"metadata must be KEY=VALUE, got {value!r}.")
        key, item = value.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError("metadata key cannot be empty.")
        metadata[key] = item
    return metadata


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-tasks", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--json", default=None)
    parser.add_argument("--audited-tasks-jsonl", default=None)
    parser.add_argument("--requeue-suggestions-jsonl", default=None)
    parser.add_argument("--artifact-manifest", default=None)
    parser.add_argument("--registry", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--version", default=None)
    parser.add_argument("--metadata", action="append", default=[])
    parser.add_argument("--compact-json", action="store_true")
    args = parser.parse_args(argv)
    payload = run(
        input_tasks_path=args.input_tasks,
        output_dir=args.output_dir,
        report_json_path=args.json,
        audited_tasks_path=args.audited_tasks_jsonl,
        requeue_suggestions_path=args.requeue_suggestions_jsonl,
        artifact_manifest_path=args.artifact_manifest,
        registry_path=args.registry,
        name=args.name,
        version=args.version,
        metadata=_parse_metadata(args.metadata or ()),
        compact_json=bool(args.compact_json),
    )
    summary = payload["summary"]
    print(
        "world_model_rule_input_plan_audit_ok "
        f"status={payload['status']} "
        f"tasks={summary['task_count']} "
        f"findings={summary['finding_count']} "
        f"requeue={summary['requeue_suggestion_count']}"
    )


if __name__ == "__main__":
    main()
