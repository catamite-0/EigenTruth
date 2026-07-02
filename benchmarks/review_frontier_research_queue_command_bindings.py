"""Review and approve frontier command bindings before execution.

This workflow consumes a ``frontier_research_queue_bound_command_plan``, the
command bindings used to build it, and optional explicit review decisions. It
writes a review report plus an updated command-bindings JSON where only
mechanically clean entries with explicit reviewer approval are marked
``approved``.

The workflow never executes commands, never fetches evidence, and never treats
bindings, local artifacts, or child reports as verifier evidence.
"""

from __future__ import annotations

import argparse
import json
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

from benchmarks.bind_frontier_research_queue_command_plan import (  # noqa: E402
    WORKFLOW as BOUND_PLAN_WORKFLOW,
)
from eigentruth.json_utils import strict_json_dumps  # noqa: E402
from eigentruth.registry import ArtifactRegistry, build_artifact_manifest  # noqa: E402

WORKFLOW = "frontier_research_queue_command_binding_review"
BINDINGS_WORKFLOW = "frontier_research_queue_command_bindings"
APPROVED_DECISIONS = {"approve", "approved", "reviewed"}
BLOCKING_DECISIONS = {"block", "blocked", "reject", "rejected"}
DEFER_DECISIONS = {"defer", "needs_review", "needs-more-evidence", "needs_more_evidence", "pending"}
RESERVED_FIELDS = {
    "answer",
    "answers",
    "is_false",
    "label",
    "labels",
    "model_answer",
    "score_label",
}


def review_frontier_research_queue_command_bindings(
    *,
    bound_command_plan: str | Path | Mapping[str, Any],
    base_bindings: str | Path | Mapping[str, Any],
    output_dir: str | Path,
    review_decisions: str | Path | Sequence[Mapping[str, Any]] | None = None,
    json_path: str | Path | None = None,
    approved_bindings_path: str | Path | None = None,
    review_template_path: str | Path | None = None,
    review_records_path: str | Path | None = None,
    artifact_manifest_path: str | Path | None = None,
    registry_path: str | Path | None = None,
    name: str | None = None,
    version: str | None = None,
    compact_json: bool = False,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Audit bound commands and stage approved command bindings."""
    if registry_path is not None and (not name or not version):
        raise ValueError("registry_path requires name and version.")
    plan_path, plan = _load_mapping_source(bound_command_plan)
    bindings_path, bindings = _load_mapping_source(base_bindings)
    if plan.get("workflow") != BOUND_PLAN_WORKFLOW:
        raise ValueError(f"bound_command_plan must have workflow={BOUND_PLAN_WORKFLOW!r}.")
    if bindings.get("workflow") != BINDINGS_WORKFLOW:
        raise ValueError(f"base_bindings must have workflow={BINDINGS_WORKFLOW!r}.")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report_path = Path(json_path or output / "frontier-command-binding-review.json")
    approved_path = Path(
        approved_bindings_path or output / "frontier-research-command-bindings.json"
    )
    template_path = Path(review_template_path or output / "command-binding-review-template.jsonl")
    records_path = Path(review_records_path or output / "command-binding-review-records.jsonl")
    manifest_path = Path(artifact_manifest_path or output / "artifact-manifest.json")

    decisions_path, decisions = _load_review_decisions(review_decisions)
    entries = tuple(_mapping_sequence(plan.get("entries", ())))
    bindings_by_action = _bindings_by_action(bindings)
    decisions_by_action = _decisions_by_action(decisions)
    templates = tuple(
        _review_template(entry, index=index)
        for index, entry in enumerate(entries, start=1)
    )
    records = tuple(
        _review_record(
            entry,
            bindings_by_action.get(str(entry.get("action_id") or ""), {}),
            decisions_by_action.get(str(entry.get("action_id") or "")),
        )
        for entry in entries
    )
    approved_bindings, apply_summary = _approved_bindings(bindings, records)
    summary = _summary(
        entries=entries,
        decisions=decisions,
        records=records,
        apply_summary=apply_summary,
    )
    status = _status(summary)
    payload = {
        "schema_version": 1,
        "workflow": WORKFLOW,
        "status": status,
        "scope": (
            "Review gate for frontier command bindings. The output may mark "
            "clean entries approved, but it does not execute commands and does "
            "not convert command bindings into verifier or release evidence."
        ),
        "source": {
            "bound_command_plan": None if plan_path is None else str(plan_path),
            "bound_plan_status": plan.get("status"),
            "base_bindings": None if bindings_path is None else str(bindings_path),
            "base_bindings_workflow": bindings.get("workflow"),
            "review_decisions": None if decisions_path is None else str(decisions_path),
        },
        "label_usage": {
            "labels_used_for_command_binding_review": False,
            "labels_allowed_in_review_decisions": False,
            "model_answers_allowed_in_review_decisions": False,
            "approved_bindings_are_verifier_evidence": False,
            "review_executes_commands": False,
        },
        "config": {
            "approved_decisions": tuple(sorted(APPROVED_DECISIONS)),
            "blocking_decisions": tuple(sorted(BLOCKING_DECISIONS)),
            "defer_decisions": tuple(sorted(DEFER_DECISIONS)),
        },
        "summary": summary,
        "paths": {
            "report": str(report_path),
            "approved_bindings": str(approved_path),
            "review_template": str(template_path),
            "review_records": str(records_path),
            "artifact_manifest": str(manifest_path),
        },
        "review_template": templates,
        "records": records,
        "approved_bindings": approved_bindings,
        "metadata": dict(metadata or {}),
    }

    _write_json(report_path, payload, compact=compact_json)
    _write_json(approved_path, approved_bindings, compact=compact_json)
    _write_jsonl(template_path, templates, compact=compact_json)
    _write_jsonl(records_path, records, compact=compact_json)
    manifest = _write_manifest(
        manifest_path=manifest_path,
        report_path=report_path,
        approved_bindings_path=approved_path,
        template_path=template_path,
        records_path=records_path,
        plan_path=plan_path,
        base_bindings_path=bindings_path,
        decisions_path=decisions_path,
        payload=payload,
        metadata=metadata or {},
        compact=compact_json,
    )
    if registry_path is not None:
        assert name is not None and version is not None
        ArtifactRegistry.load_json(registry_path).record_report(
            name=name,
            version=version,
            path=report_path,
            metadata={
                "workflow": WORKFLOW,
                "status": status,
                "artifact_manifest": str(manifest_path),
                "entry_count": summary["entry_count"],
                "approved_entry_count": summary["approved_entry_count"],
                "blocked_entry_count": summary["blocked_entry_count"],
                "pending_review_count": summary["pending_review_count"],
                "manifest_summary": manifest.get("summary", {}),
                **dict(metadata or {}),
            },
        ).save_json()
    return payload


def _review_template(entry: Mapping[str, Any], *, index: int) -> dict[str, Any]:
    action_id = str(entry.get("action_id") or "")
    return {
        "review_id": f"frontier-command-binding-review-{index}",
        "action_id": action_id,
        "entry_id": str(entry.get("entry_id") or action_id),
        "title": str(entry.get("title") or action_id),
        "decision": "needs_review",
        "reviewer": "",
        "reviewed_at": "",
        "not_verifier_evidence": True,
        "required_checks": (
            "bound_command_plan_entry_ready",
            "no_unbound_placeholders",
            "command_validation_clean",
            "inputs_do_not_include_reserved_label_fields",
            "local_artifact_inputs_reviewed_when_present",
        ),
        "command_count": len(_string_tuple(entry.get("bound_commands", ()))),
        "required_inputs": _string_tuple(entry.get("required_inputs", ())),
        "unbound_inputs": _string_tuple(entry.get("unbound_inputs", ())),
    }


def _review_record(
    entry: Mapping[str, Any],
    binding: Mapping[str, Any],
    decision: Mapping[str, Any] | None,
) -> dict[str, Any]:
    action_id = str(entry.get("action_id") or "")
    failures = list(_mechanical_failures(entry, binding))
    decision_status = _decision_status(decision)
    if decision is None:
        failures.append("missing_review_decision")
    elif decision_status == "approved":
        if not _clean(decision.get("reviewer")):
            failures.append("missing_reviewer")
        if _bool_true(decision.get("not_verifier_evidence")) is not True:
            failures.append("not_verifier_evidence_not_true")
        reserved_decision_fields = tuple(
            sorted(key for key in decision if str(key) in RESERVED_FIELDS)
        )
        if reserved_decision_fields:
            failures.append("reserved_review_decision_fields")
    elif decision_status == "blocked":
        failures.append("review_decision_blocked")
    else:
        failures.append("review_decision_pending")

    status = "approved" if not failures else "blocked"
    return {
        "action_id": action_id,
        "entry_id": str(entry.get("entry_id") or action_id),
        "title": str(entry.get("title") or action_id),
        "status": status,
        "decision": decision_status,
        "reviewer": "" if decision is None else _clean(decision.get("reviewer")),
        "reviewed_at": "" if decision is None else _clean(decision.get("reviewed_at")),
        "not_verifier_evidence": False
        if decision is None
        else _bool_true(decision.get("not_verifier_evidence")),
        "failures": tuple(dict.fromkeys(failures)),
        "command_status": str(entry.get("command_status") or ""),
        "command_count": len(_string_tuple(entry.get("bound_commands", ()))),
        "unbound_inputs": _string_tuple(entry.get("unbound_inputs", ())),
        "binding_review_status_before": str(binding.get("review_status") or ""),
    }


def _mechanical_failures(
    entry: Mapping[str, Any],
    binding: Mapping[str, Any],
) -> tuple[str, ...]:
    failures: list[str] = []
    if str(entry.get("command_status") or "") != "ready":
        failures.append("bound_entry_not_ready")
    if _string_tuple(entry.get("unbound_inputs", ())):
        failures.append("unbound_inputs_present")
    commands = _string_tuple(entry.get("bound_commands", ()))
    if not commands:
        failures.append("missing_bound_commands")
    if any("..." in command for command in commands):
        failures.append("unbound_command_placeholders")
    validation = _mapping(entry.get("command_validation"))
    if _int_or_zero(validation.get("issue_count")) > 0:
        failures.append("command_validation_issues")
    if _reserved_paths(_mapping(entry.get("bound_inputs"))):
        failures.append("reserved_bound_input_fields")
    if _reserved_paths(_mapping(binding.get("inputs"))):
        failures.append("reserved_binding_input_fields")
    artifact_reviews = tuple(_mapping_sequence(binding.get("artifact_input_reviews", ())))
    for review in artifact_reviews:
        review_status = _clean(review.get("review_status")).lower()
        if review_status not in {"approved", "reviewed"}:
            failures.append("artifact_input_review_not_approved")
        if _bool_true(review.get("not_verifier_evidence")) is not True:
            failures.append("artifact_input_not_marked_non_evidence")
        if _reserved_paths(review):
            failures.append("reserved_artifact_input_review_fields")
    return tuple(dict.fromkeys(failures))


def _approved_bindings(
    base: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, int]]:
    approved = dict(base)
    bindings = {str(key): dict(_mapping(value)) for key, value in _mapping(base.get("bindings")).items()}
    approved_count = 0
    missing_binding_count = 0
    for record in records:
        if record.get("status") != "approved":
            continue
        action_id = str(record.get("action_id") or "")
        if action_id not in bindings:
            missing_binding_count += 1
            continue
        binding = dict(bindings[action_id])
        binding["review_status"] = "approved"
        binding["reviewer"] = str(record.get("reviewer") or "")
        binding["reviewed_at"] = str(record.get("reviewed_at") or "")
        binding["command_binding_review"] = {
            "workflow": WORKFLOW,
            "status": "approved",
            "not_verifier_evidence": True,
        }
        bindings[action_id] = binding
        approved_count += 1
    approved["workflow"] = BINDINGS_WORKFLOW
    approved["status"] = "approved" if approved_count else "needs_review"
    approved["bindings"] = bindings
    approved["generated_by"] = WORKFLOW
    return approved, {
        "approved_binding_count": approved_count,
        "missing_binding_count": missing_binding_count,
    }


def _summary(
    *,
    entries: Sequence[Mapping[str, Any]],
    decisions: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
    apply_summary: Mapping[str, Any],
) -> dict[str, Any]:
    status_counts = Counter(str(record.get("status") or "") for record in records)
    failure_counts: Counter[str] = Counter()
    for record in records:
        for failure in _string_tuple(record.get("failures", ())):
            failure_counts[failure] += 1
    return {
        "entry_count": len(entries),
        "review_decision_count": len(decisions),
        "approved_entry_count": status_counts.get("approved", 0),
        "blocked_entry_count": status_counts.get("blocked", 0),
        "pending_review_count": failure_counts.get("missing_review_decision", 0)
        + failure_counts.get("review_decision_pending", 0),
        "approved_binding_count": _int_or_zero(apply_summary.get("approved_binding_count")),
        "missing_binding_count": _int_or_zero(apply_summary.get("missing_binding_count")),
        "record_status_counts": _sorted_counter(status_counts),
        "failure_counts": _sorted_counter(failure_counts),
    }


def _status(summary: Mapping[str, Any]) -> str:
    if _int_or_zero(summary.get("entry_count")) == 0:
        return "empty"
    if _int_or_zero(summary.get("blocked_entry_count")) > 0:
        return "needs_review"
    if _int_or_zero(summary.get("missing_binding_count")) > 0:
        return "needs_review"
    if _int_or_zero(summary.get("approved_entry_count")) < _int_or_zero(
        summary.get("entry_count")
    ):
        return "needs_review"
    return "ready_for_execution"


def _load_review_decisions(
    source: str | Path | Sequence[Mapping[str, Any]] | None,
) -> tuple[Path | None, tuple[Mapping[str, Any], ...]]:
    if source is None:
        return None, ()
    if isinstance(source, (str, Path)):
        path = Path(source)
        return path, _load_jsonl_mappings(path)
    if isinstance(source, Sequence) and not isinstance(source, (bytes, bytearray, str)):
        return None, tuple(item for item in source if isinstance(item, Mapping))
    raise ValueError("review_decisions must be a JSONL path or sequence of objects.")


def _decisions_by_action(decisions: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for decision in decisions:
        action_id = str(decision.get("action_id") or "")
        if action_id and action_id not in result:
            result[action_id] = decision
    return result


def _decision_status(decision: Mapping[str, Any] | None) -> str:
    if decision is None:
        return "missing"
    value = _clean(decision.get("decision")).lower()
    if value in APPROVED_DECISIONS:
        return "approved"
    if value in BLOCKING_DECISIONS:
        return "blocked"
    if value in DEFER_DECISIONS or not value:
        return "pending"
    return "pending"


def _reserved_paths(value: Any, *, prefix: str = "") -> tuple[str, ...]:
    paths = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            path = f"{prefix}.{key_text}" if prefix else key_text
            if key_text in RESERVED_FIELDS:
                paths.append(path)
            paths.extend(_reserved_paths(child, prefix=path))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            path = f"{prefix}[{index}]" if prefix else f"[{index}]"
            paths.extend(_reserved_paths(child, prefix=path))
    return tuple(paths)


def _bindings_by_action(payload: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    raw = payload.get("bindings", {})
    if not isinstance(raw, Mapping):
        return {}
    return {str(key): _mapping(value) for key, value in raw.items()}


def _write_manifest(
    *,
    manifest_path: Path,
    report_path: Path,
    approved_bindings_path: Path,
    template_path: Path,
    records_path: Path,
    plan_path: Path | None,
    base_bindings_path: Path | None,
    decisions_path: Path | None,
    payload: Mapping[str, Any],
    metadata: Mapping[str, Any],
    compact: bool,
) -> Mapping[str, Any]:
    manifest = build_artifact_manifest(
        {
            "frontier_command_binding_review": report_path,
            "approved_command_bindings": approved_bindings_path,
            "review_template": template_path,
            "review_records": records_path,
            "bound_command_plan": plan_path,
            "base_bindings": base_bindings_path,
            "review_decisions": decisions_path,
        },
        root=manifest_path.parent,
        metadata={
            "runner": "review_frontier_research_queue_command_bindings",
            "workflow": WORKFLOW,
            "status": payload.get("status"),
            "entry_count": _nested(payload, "summary", "entry_count"),
            "approved_entry_count": _nested(payload, "summary", "approved_entry_count"),
            "blocked_entry_count": _nested(payload, "summary", "blocked_entry_count"),
            **dict(metadata),
        },
    )
    _write_json(manifest_path, manifest, compact=compact)
    return manifest


def _load_mapping_source(source: str | Path | Mapping[str, Any]) -> tuple[Path | None, dict[str, Any]]:
    if isinstance(source, Mapping):
        return None, dict(source)
    path = Path(source)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must contain a JSON object.")
    return path, dict(payload)


def _load_jsonl_mappings(path: str | Path) -> tuple[Mapping[str, Any], ...]:
    rows = []
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, Mapping):
            raise ValueError(f"{path}:{line_no} must contain a JSON object.")
        rows.append({str(key): value for key, value in row.items()})
    return tuple(rows)


def _bool_true(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return False


def _clean(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _mapping_sequence(value: Any) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return ()
    return tuple(item for item in value if isinstance(item, Mapping))


def _string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,) if value else ()
    if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray)):
        return (str(value),) if str(value) else ()
    return tuple(str(item) for item in value if str(item))


def _int_or_zero(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _nested(payload: Mapping[str, Any], *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _sorted_counter(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted((str(key), int(value)) for key, value in counter.items() if str(key)))


def _write_json(path: str | Path, payload: Mapping[str, Any], *, compact: bool = False) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    text = (
        strict_json_dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
        if compact
        else strict_json_dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    output.write_text(text, encoding="utf-8")


def _write_jsonl(path: str | Path, rows: Sequence[Mapping[str, Any]], *, compact: bool = False) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if compact:
        text = "".join(strict_json_dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows)
    else:
        text = "".join(strict_json_dumps(row, sort_keys=True) + "\n" for row in rows)
    output.write_text(text, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bound-command-plan", required=True)
    parser.add_argument("--base-bindings", required=True)
    parser.add_argument("--review-decisions", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--json", default=None)
    parser.add_argument("--approved-bindings", default=None)
    parser.add_argument("--review-template", default=None)
    parser.add_argument("--review-records", default=None)
    parser.add_argument("--artifact-manifest", default=None)
    parser.add_argument("--registry", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--version", default=None)
    parser.add_argument("--compact-json", action="store_true")
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    payload = review_frontier_research_queue_command_bindings(
        bound_command_plan=args.bound_command_plan,
        base_bindings=args.base_bindings,
        review_decisions=args.review_decisions,
        output_dir=args.output_dir,
        json_path=args.json,
        approved_bindings_path=args.approved_bindings,
        review_template_path=args.review_template,
        review_records_path=args.review_records,
        artifact_manifest_path=args.artifact_manifest,
        registry_path=args.registry,
        name=args.name,
        version=args.version,
        compact_json=bool(args.compact_json),
    )
    if args.json is None:
        print(strict_json_dumps(payload, indent=2, sort_keys=True))
    return payload


def main(argv: Sequence[str] | None = None) -> None:
    run(build_parser().parse_args(argv))


if __name__ == "__main__":  # pragma: no cover
    main()
