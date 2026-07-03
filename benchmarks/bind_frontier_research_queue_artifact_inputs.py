"""Bind reviewed local artifact inputs into frontier command bindings.

This workflow consumes ``frontier_research_queue_input_collection_plan`` local
artifact requests, a reviewed JSONL sidecar of artifact paths, and an existing
``frontier_research_queue_command_bindings`` file. It writes an updated command
bindings file with approved artifact paths inserted into matching command
placeholders.

The output remains a binding artifact. It does not execute commands, approve the
whole command binding, fetch evidence, or treat local artifacts as verifier
evidence.
"""

from __future__ import annotations

import argparse
import json
import shlex
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

from benchmarks.plan_frontier_research_queue_input_collection import (  # noqa: E402
    WORKFLOW as COLLECTION_WORKFLOW,
)
from eigentruth.json_utils import strict_json_dumps  # noqa: E402
from eigentruth.registry import ArtifactRegistry, build_artifact_manifest  # noqa: E402

WORKFLOW = "frontier_research_queue_artifact_input_binding_staging"
BINDINGS_WORKFLOW = "frontier_research_queue_command_bindings"
APPROVED_REVIEW_STATUSES = {"approved", "reviewed"}
RESERVED_FIELDS = {
    "answer",
    "answers",
    "is_false",
    "label",
    "labels",
    "model_answer",
    "score_label",
}


def bind_frontier_research_queue_artifact_inputs(
    *,
    input_collection_plan: str | Path | Mapping[str, Any],
    base_bindings: str | Path | Mapping[str, Any],
    output_dir: str | Path,
    artifact_bindings: str | Path | None = None,
    json_path: str | Path | None = None,
    bindings_json_path: str | Path | None = None,
    template_jsonl_path: str | Path | None = None,
    records_jsonl_path: str | Path | None = None,
    artifact_manifest_path: str | Path | None = None,
    registry_path: str | Path | None = None,
    name: str | None = None,
    version: str | None = None,
    require_existing_artifacts: bool = True,
    compact_json: bool = False,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Stage approved local artifact input rows into command bindings."""
    if registry_path is not None and (not name or not version):
        raise ValueError("registry_path requires name and version.")
    if not isinstance(require_existing_artifacts, bool):
        raise ValueError("require_existing_artifacts must be a bool.")

    collection_path, collection_plan = _load_mapping_source(input_collection_plan)
    bindings_path, base = _load_mapping_source(base_bindings)
    if collection_plan.get("workflow") != COLLECTION_WORKFLOW:
        raise ValueError(
            f"input_collection_plan must have workflow={COLLECTION_WORKFLOW!r}."
        )
    if base.get("workflow") != BINDINGS_WORKFLOW:
        raise ValueError(f"base_bindings must have workflow={BINDINGS_WORKFLOW!r}.")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report_path = Path(json_path or output / "artifact-input-binding-staging.json")
    command_bindings_path = Path(
        bindings_json_path or output / "frontier-research-command-bindings.json"
    )
    template_path = Path(template_jsonl_path or output / "artifact-input-review-template.jsonl")
    records_path = Path(records_jsonl_path or output / "artifact-input-binding-records.jsonl")
    manifest_path = Path(artifact_manifest_path or output / "artifact-manifest.json")

    binding_rows_path = None if artifact_bindings is None else Path(artifact_bindings)
    binding_rows = () if binding_rows_path is None else _load_jsonl_mappings(binding_rows_path)
    row_root = None if binding_rows_path is None else binding_rows_path.parent
    requests = tuple(_artifact_requests(collection_plan))
    templates = tuple(_review_template(request, index=index) for index, request in enumerate(requests, start=1))
    records = tuple(
        _artifact_binding_records(
            requests,
            binding_rows,
            row_root=row_root,
            require_existing_artifacts=require_existing_artifacts,
        )
    )
    updated_bindings, apply_summary = _updated_bindings(base, records)
    summary = _summary(
        requests=requests,
        templates=templates,
        binding_rows=binding_rows,
        records=records,
        apply_summary=apply_summary,
    )
    status = _status(summary)
    payload = {
        "schema_version": 1,
        "workflow": WORKFLOW,
        "status": status,
        "scope": (
            "Stages explicitly reviewed local artifact paths into frontier "
            "command bindings. These bindings remain non-evidence and still "
            "need normal command-binding review before execution."
        ),
        "source": {
            "input_collection_plan": None if collection_path is None else str(collection_path),
            "input_collection_workflow": collection_plan.get("workflow"),
            "base_bindings": None if bindings_path is None else str(bindings_path),
            "base_bindings_workflow": base.get("workflow"),
            "artifact_bindings": None if binding_rows_path is None else str(binding_rows_path),
        },
        "label_usage": {
            "labels_used_for_artifact_binding": False,
            "labels_allowed_in_artifact_bindings": False,
            "model_answers_allowed_in_artifact_bindings": False,
            "artifact_inputs_are_verifier_evidence": False,
            "stage_approves_command_bindings": False,
            "stage_executes_commands": False,
        },
        "config": {
            "require_existing_artifacts": bool(require_existing_artifacts),
            "approved_review_statuses": tuple(sorted(APPROVED_REVIEW_STATUSES)),
        },
        "summary": summary,
        "paths": {
            "report": str(report_path),
            "command_bindings": str(command_bindings_path),
            "artifact_input_review_template": str(template_path),
            "artifact_input_binding_records": str(records_path),
            "artifact_manifest": str(manifest_path),
        },
        "review_template": templates,
        "records": records,
        "updated_bindings": updated_bindings,
        "metadata": dict(metadata or {}),
    }

    _write_json(report_path, payload, compact=compact_json)
    _write_json(command_bindings_path, updated_bindings, compact=compact_json)
    _write_jsonl(template_path, templates, compact=compact_json)
    _write_jsonl(records_path, records, compact=compact_json)
    manifest = _write_manifest(
        manifest_path=manifest_path,
        output_path=report_path,
        command_bindings_path=command_bindings_path,
        template_path=template_path,
        records_path=records_path,
        collection_path=collection_path,
        base_bindings_path=bindings_path,
        artifact_bindings_path=binding_rows_path,
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
                "artifact_request_count": summary["artifact_request_count"],
                "approved_artifact_input_count": summary["approved_artifact_input_count"],
                "applied_input_count": summary["applied_input_count"],
                "applied_placeholder_count": summary["applied_placeholder_count"],
                "blocked_artifact_input_count": summary["blocked_artifact_input_count"],
                "manifest_summary": manifest.get("summary", {}),
                **dict(metadata or {}),
            },
        ).save_json()
    return payload


def _artifact_requests(collection_plan: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    return tuple(
        request
        for request in _mapping_sequence(collection_plan.get("collection_requests", ()))
        if str(request.get("input_category") or "") == "local_artifact"
    )


def _review_template(request: Mapping[str, Any], *, index: int) -> dict[str, Any]:
    skeleton = dict(_mapping(request.get("recommended_review_skeleton")))
    for field in _string_tuple(request.get("required_review_fields", ())):
        if field == "not_verifier_evidence":
            skeleton.setdefault(field, True)
        else:
            skeleton.setdefault(field, "")
    return {
        "template_id": f"artifact-input-review-{index}",
        "collection_request_id": str(request.get("request_id") or ""),
        "action_id": str(request.get("action_id") or ""),
        "input_name": str(request.get("input_name") or ""),
        "target_flag": str(request.get("target_flag") or ""),
        "artifact_family": str(request.get("artifact_family") or ""),
        "required_review_fields": _string_tuple(request.get("required_review_fields", ())),
        **skeleton,
    }


def _artifact_binding_records(
    requests: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    *,
    row_root: Path | None,
    require_existing_artifacts: bool,
) -> tuple[dict[str, Any], ...]:
    by_key: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_key.setdefault(_row_key(row), []).append(row)

    records = []
    used_keys: set[str] = set()
    for request in requests:
        key = _request_key(request)
        used_keys.add(key)
        matches = by_key.get(key, ())
        if len(matches) > 1:
            records.append(_blocked_record(request, skip_reasons=("duplicate_artifact_binding",)))
            continue
        if not matches:
            records.append(_blocked_record(request, skip_reasons=("pending_review",)))
            continue
        records.append(
            _record_from_row(
                request,
                matches[0],
                row_root=row_root,
                require_existing_artifacts=require_existing_artifacts,
            )
        )

    for key, matches in sorted(by_key.items()):
        if key in used_keys:
            continue
        for row in matches:
            records.append({
                "collection_request_id": str(row.get("collection_request_id") or ""),
                "action_id": str(row.get("action_id") or ""),
                "input_name": str(row.get("input_name") or ""),
                "status": "blocked",
                "skip_reasons": ("unknown_collection_request",),
                "not_verifier_evidence": _bool_true(row.get("not_verifier_evidence")),
            })
    return tuple(records)


def _record_from_row(
    request: Mapping[str, Any],
    row: Mapping[str, Any],
    *,
    row_root: Path | None,
    require_existing_artifacts: bool,
) -> dict[str, Any]:
    skip_reasons: list[str] = []
    reserved = tuple(sorted(key for key in row if str(key) in RESERVED_FIELDS))
    if reserved:
        skip_reasons.append("reserved_review_fields")
    review_status = str(row.get("review_status") or "").strip().lower()
    if review_status not in APPROVED_REVIEW_STATUSES:
        skip_reasons.append("review_not_approved")
    not_verifier_evidence = _bool_true(row.get("not_verifier_evidence"))
    if not_verifier_evidence is not True:
        skip_reasons.append("not_verifier_evidence_not_true")

    missing_required = []
    for field in _string_tuple(request.get("required_review_fields", ())):
        if field == "not_verifier_evidence":
            continue
        if not _clean(row.get(field)):
            missing_required.append(field)
    if missing_required:
        skip_reasons.append("missing_required_review_fields")

    expected_label_use = _clean(
        _mapping(request.get("recommended_review_skeleton")).get("allowed_label_use")
    )
    allowed_label_use = _clean(row.get("allowed_label_use"))
    if expected_label_use and allowed_label_use != expected_label_use:
        skip_reasons.append("unexpected_allowed_label_use")

    artifact_path = _clean(row.get("artifact_path"))
    artifact_manifest = _clean(row.get("artifact_manifest"))
    resolved_artifact = _resolve_local_path(artifact_path, source_root=row_root)
    resolved_manifest = _resolve_local_path(artifact_manifest, source_root=row_root)
    if require_existing_artifacts:
        if not artifact_path or resolved_artifact is None or not resolved_artifact.exists():
            skip_reasons.append("artifact_path_not_materialized")
        if artifact_manifest and resolved_manifest is not None and not resolved_manifest.exists():
            skip_reasons.append("artifact_manifest_not_materialized")

    status = "ready" if not skip_reasons else "blocked"
    return {
        "collection_request_id": str(request.get("request_id") or ""),
        "action_id": str(request.get("action_id") or ""),
        "input_name": str(request.get("input_name") or ""),
        "target_flag": str(request.get("target_flag") or ""),
        "artifact_family": str(request.get("artifact_family") or ""),
        "status": status,
        "skip_reasons": tuple(dict.fromkeys(skip_reasons)),
        "artifact_path": "" if resolved_artifact is None else str(resolved_artifact),
        "artifact_manifest": "" if resolved_manifest is None else str(resolved_manifest),
        "source_workflow": _clean(row.get("source_workflow")),
        "allowed_label_use": allowed_label_use,
        "review_status": review_status,
        "reviewer": _clean(row.get("reviewer")),
        "reviewed_at": _clean(row.get("reviewed_at")),
        "not_verifier_evidence": not_verifier_evidence,
        "reserved_review_fields": reserved,
    }


def _blocked_record(
    request: Mapping[str, Any],
    *,
    skip_reasons: Sequence[str],
) -> dict[str, Any]:
    return {
        "collection_request_id": str(request.get("request_id") or ""),
        "action_id": str(request.get("action_id") or ""),
        "input_name": str(request.get("input_name") or ""),
        "target_flag": str(request.get("target_flag") or ""),
        "artifact_family": str(request.get("artifact_family") or ""),
        "status": "blocked",
        "skip_reasons": tuple(skip_reasons),
        "artifact_path": "",
        "artifact_manifest": "",
        "source_workflow": "",
        "allowed_label_use": "",
        "review_status": "",
        "reviewer": "",
        "reviewed_at": "",
        "not_verifier_evidence": False,
        "reserved_review_fields": (),
    }


def _updated_bindings(
    base: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    updated = dict(base)
    inputs = dict(_mapping(base.get("inputs")))
    bindings = {str(key): dict(_mapping(value)) for key, value in _mapping(base.get("bindings")).items()}
    ready_records = tuple(record for record in records if record.get("status") == "ready")
    records_by_action: dict[str, list[Mapping[str, Any]]] = {}
    for record in ready_records:
        records_by_action.setdefault(str(record.get("action_id") or ""), []).append(record)

    applied_inputs = 0
    applied_placeholders = 0
    missing_action_bindings = 0
    unapplied_records = 0
    for action_id, action_records in sorted(records_by_action.items()):
        if not action_id:
            unapplied_records += len(action_records)
            continue
        binding = bindings.get(action_id)
        if binding is None:
            missing_action_bindings += 1
            unapplied_records += len(action_records)
            continue
        replacements = {}
        artifact_reviews = list(_mapping_sequence(binding.get("artifact_input_reviews", ())))
        entry_inputs = dict(_mapping(binding.get("inputs")))
        for record in action_records:
            input_name = str(record.get("input_name") or "")
            target_flag = str(record.get("target_flag") or "")
            artifact_path = str(record.get("artifact_path") or "")
            if not input_name or not target_flag or not artifact_path:
                unapplied_records += 1
                continue
            input_value = _input_value(record)
            inputs[input_name] = input_value
            entry_inputs[input_name] = input_value
            replacements[target_flag] = artifact_path
            artifact_reviews.append(_artifact_input_review(record))
            applied_inputs += 1
        command_result = _bind_artifact_placeholders(
            _string_tuple(binding.get("bound_commands", ())),
            replacements,
        )
        applied_placeholders += command_result["applied_placeholder_count"]
        binding["inputs"] = entry_inputs
        binding["artifact_input_reviews"] = tuple(artifact_reviews)
        if command_result["commands"]:
            binding["bound_commands"] = command_result["commands"]
        binding.setdefault("review_status", "needs_review")
        bindings[action_id] = binding

    updated["workflow"] = BINDINGS_WORKFLOW
    updated["status"] = "needs_review"
    updated["inputs"] = inputs
    updated["bindings"] = bindings
    updated["generated_by"] = WORKFLOW
    return updated, {
        "applied_input_count": applied_inputs,
        "applied_placeholder_count": applied_placeholders,
        "missing_action_binding_count": missing_action_bindings,
        "unapplied_artifact_input_count": unapplied_records,
    }


def _bind_artifact_placeholders(
    commands: Sequence[str],
    replacements: Mapping[str, str],
) -> dict[str, Any]:
    if not commands:
        return {"commands": (), "applied_placeholder_count": 0}
    applied = 0
    updated_commands = []
    for command in commands:
        try:
            tokens = shlex.split(str(command))
        except ValueError:
            tokens = str(command).split()
        for index, token in enumerate(tokens):
            if token != "..." or index == 0:
                continue
            replacement = replacements.get(tokens[index - 1])
            if replacement:
                tokens[index] = replacement
                applied += 1
        updated_commands.append(shlex.join(tokens))
    return {
        "commands": tuple(updated_commands),
        "applied_placeholder_count": applied,
    }


def _input_value(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "path": str(record.get("artifact_path") or ""),
        "artifact_manifest": str(record.get("artifact_manifest") or ""),
        "source_workflow": str(record.get("source_workflow") or ""),
        "allowed_label_use": str(record.get("allowed_label_use") or ""),
        "review_status": str(record.get("review_status") or ""),
        "reviewer": str(record.get("reviewer") or ""),
        "reviewed_at": str(record.get("reviewed_at") or ""),
        "not_verifier_evidence": record.get("not_verifier_evidence") is True,
    }


def _artifact_input_review(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "collection_request_id": str(record.get("collection_request_id") or ""),
        "input_name": str(record.get("input_name") or ""),
        "target_flag": str(record.get("target_flag") or ""),
        "artifact_path": str(record.get("artifact_path") or ""),
        "review_status": str(record.get("review_status") or ""),
        "not_verifier_evidence": record.get("not_verifier_evidence") is True,
    }


def _summary(
    *,
    requests: Sequence[Mapping[str, Any]],
    templates: Sequence[Mapping[str, Any]],
    binding_rows: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
    apply_summary: Mapping[str, Any],
) -> dict[str, Any]:
    status_counts = Counter(str(record.get("status") or "") for record in records)
    artifact_family_counts = Counter(str(request.get("artifact_family") or "") for request in requests)
    skip_counts: Counter[str] = Counter()
    for record in records:
        for reason in _string_tuple(record.get("skip_reasons", ())):
            skip_counts[reason] += 1
    return {
        "artifact_request_count": len(requests),
        "review_template_count": len(templates),
        "artifact_binding_row_count": len(binding_rows),
        "approved_artifact_input_count": status_counts.get("ready", 0),
        "blocked_artifact_input_count": sum(
            count for status, count in status_counts.items() if status != "ready"
        ),
        "applied_input_count": _int_or_zero(apply_summary.get("applied_input_count")),
        "applied_placeholder_count": _int_or_zero(
            apply_summary.get("applied_placeholder_count")
        ),
        "missing_action_binding_count": _int_or_zero(
            apply_summary.get("missing_action_binding_count")
        ),
        "unapplied_artifact_input_count": _int_or_zero(
            apply_summary.get("unapplied_artifact_input_count")
        ),
        "artifact_family_counts": _sorted_counter(artifact_family_counts),
        "record_status_counts": _sorted_counter(status_counts),
        "skip_reason_counts": _sorted_counter(skip_counts),
    }


def _status(summary: Mapping[str, Any]) -> str:
    if _int_or_zero(summary.get("artifact_request_count")) == 0:
        return "empty"
    if _int_or_zero(summary.get("blocked_artifact_input_count")) > 0:
        return "needs_review"
    if _int_or_zero(summary.get("missing_action_binding_count")) > 0:
        return "needs_review"
    if _int_or_zero(summary.get("unapplied_artifact_input_count")) > 0:
        return "needs_review"
    if _int_or_zero(summary.get("applied_input_count")) < _int_or_zero(
        summary.get("artifact_request_count")
    ):
        return "needs_review"
    return "ready_for_binding_review"


def _request_key(request: Mapping[str, Any]) -> str:
    request_id = str(request.get("request_id") or "")
    if request_id:
        return f"request:{request_id}"
    return f"action:{request.get('action_id')}:{request.get('input_name')}"


def _row_key(row: Mapping[str, Any]) -> str:
    request_id = str(row.get("collection_request_id") or row.get("request_id") or "")
    if request_id:
        return f"request:{request_id}"
    return f"action:{row.get('action_id')}:{row.get('input_name')}"


def _write_manifest(
    *,
    manifest_path: Path,
    output_path: Path,
    command_bindings_path: Path,
    template_path: Path,
    records_path: Path,
    collection_path: Path | None,
    base_bindings_path: Path | None,
    artifact_bindings_path: Path | None,
    payload: Mapping[str, Any],
    metadata: Mapping[str, Any],
    compact: bool,
) -> Mapping[str, Any]:
    manifest = build_artifact_manifest(
        {
            "artifact_input_binding_staging_report": output_path,
            "frontier_research_queue_command_bindings": command_bindings_path,
            "artifact_input_review_template": template_path,
            "artifact_input_binding_records": records_path,
            "input_collection_plan": collection_path,
            "base_bindings": base_bindings_path,
            "artifact_bindings": artifact_bindings_path,
        },
        root=manifest_path.parent,
        metadata={
            "runner": "bind_frontier_research_queue_artifact_inputs",
            "workflow": WORKFLOW,
            "status": payload.get("status"),
            "artifact_request_count": _nested(payload, "summary", "artifact_request_count"),
            "approved_artifact_input_count": _nested(
                payload,
                "summary",
                "approved_artifact_input_count",
            ),
            "applied_placeholder_count": _nested(
                payload,
                "summary",
                "applied_placeholder_count",
            ),
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


def _resolve_local_path(path: str, *, source_root: Path | None) -> Path | None:
    if not path:
        return None
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    if source_root is not None and (source_root / candidate).exists():
        return source_root / candidate
    if candidate.exists():
        return candidate
    return ROOT / candidate


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
    parser.add_argument("--input-collection-plan", required=True)
    parser.add_argument("--base-bindings", required=True)
    parser.add_argument("--artifact-bindings", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--json", default=None)
    parser.add_argument("--bindings-json", default=None)
    parser.add_argument("--template-jsonl", default=None)
    parser.add_argument("--records-jsonl", default=None)
    parser.add_argument("--artifact-manifest", default=None)
    parser.add_argument("--registry", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--version", default=None)
    parser.add_argument(
        "--allow-missing-artifacts",
        action="store_true",
        help="stage approved artifact paths without requiring local files to exist",
    )
    parser.add_argument("--compact-json", action="store_true")
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    payload = bind_frontier_research_queue_artifact_inputs(
        input_collection_plan=args.input_collection_plan,
        base_bindings=args.base_bindings,
        artifact_bindings=args.artifact_bindings,
        output_dir=args.output_dir,
        json_path=args.json,
        bindings_json_path=args.bindings_json,
        template_jsonl_path=args.template_jsonl,
        records_jsonl_path=args.records_jsonl,
        artifact_manifest_path=args.artifact_manifest,
        registry_path=args.registry,
        name=args.name,
        version=args.version,
        require_existing_artifacts=not bool(args.allow_missing_artifacts),
        compact_json=bool(args.compact_json),
    )
    if args.json is None:
        print(strict_json_dumps(payload, indent=2, sort_keys=True))
    return payload


def main(argv: Sequence[str] | None = None) -> None:
    run(build_parser().parse_args(argv))


if __name__ == "__main__":  # pragma: no cover
    main()
