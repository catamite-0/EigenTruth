"""Stage audited source-family URL seed sidecars into frontier command bindings.

This workflow consumes a ``frontier_research_queue_input_binding_audit`` report
plus an existing ``frontier_research_queue_command_bindings`` file. When the
``source_family_url_seeds`` sidecar is audit-ready, it binds that sidecar path
to matching ``--seeds ...`` command placeholders.

The output remains a binding artifact. It does not execute source-family
adapters, approve command bindings, fetch pages, or treat URL seeds as verifier
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

from benchmarks.audit_frontier_research_queue_input_bindings import (  # noqa: E402
    WORKFLOW as AUDIT_WORKFLOW,
)
from eigentruth.json_utils import strict_json_dumps  # noqa: E402
from eigentruth.registry import ArtifactRegistry, build_artifact_manifest  # noqa: E402

WORKFLOW = "frontier_research_queue_source_family_seed_binding_staging"
BINDINGS_WORKFLOW = "frontier_research_queue_command_bindings"
SIDECAR_KEY = "source_family_url_seeds"
INPUT_NAME = "source_family_url_seeds"
TARGET_FLAG = "--seeds"


def bind_frontier_research_queue_seed_inputs(
    *,
    input_binding_audit: str | Path | Mapping[str, Any],
    base_bindings: str | Path | Mapping[str, Any],
    output_dir: str | Path,
    json_path: str | Path | None = None,
    bindings_json_path: str | Path | None = None,
    records_jsonl_path: str | Path | None = None,
    artifact_manifest_path: str | Path | None = None,
    registry_path: str | Path | None = None,
    name: str | None = None,
    version: str | None = None,
    compact_json: bool = False,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Stage audited URL seed sidecar paths into command bindings."""
    if registry_path is not None and (not name or not version):
        raise ValueError("registry_path requires name and version.")
    audit_path, audit = _load_mapping_source(input_binding_audit)
    bindings_path, base = _load_mapping_source(base_bindings)
    if audit.get("workflow") != AUDIT_WORKFLOW:
        raise ValueError(f"input_binding_audit must have workflow={AUDIT_WORKFLOW!r}.")
    if base.get("workflow") != BINDINGS_WORKFLOW:
        raise ValueError(f"base_bindings must have workflow={BINDINGS_WORKFLOW!r}.")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report_path = Path(json_path or output / "source-family-seed-binding-staging.json")
    command_bindings_path = Path(
        bindings_json_path or output / "frontier-research-command-bindings.json"
    )
    records_path = Path(records_jsonl_path or output / "source-family-seed-binding-records.jsonl")
    manifest_path = Path(artifact_manifest_path or output / "artifact-manifest.json")

    seed_sidecar_path = _source_family_seed_sidecar_path(
        audit,
        source_root=None if audit_path is None else audit_path.parent,
    )
    seed_rows = _load_jsonl_mappings(seed_sidecar_path) if seed_sidecar_path and seed_sidecar_path.exists() else ()
    records = _seed_binding_records(
        audit,
        seed_rows=seed_rows,
        seed_sidecar_path=seed_sidecar_path,
    )
    updated_bindings, apply_summary = _updated_bindings(base, records)
    summary = _summary(
        records=records,
        apply_summary=apply_summary,
        seed_rows=seed_rows,
        seed_sidecar_path=seed_sidecar_path,
    )
    status = _status(summary)
    payload = {
        "schema_version": 1,
        "workflow": WORKFLOW,
        "status": status,
        "scope": (
            "Stages audited source-family URL seed sidecar paths into frontier "
            "command bindings. These bindings remain non-evidence and still "
            "need normal command-binding review before adapter execution."
        ),
        "source": {
            "input_binding_audit": None if audit_path is None else str(audit_path),
            "input_binding_audit_workflow": audit.get("workflow"),
            "input_binding_audit_status": audit.get("status"),
            "base_bindings": None if bindings_path is None else str(bindings_path),
            "base_bindings_workflow": base.get("workflow"),
        },
        "label_usage": {
            "labels_used_for_seed_binding": False,
            "labels_allowed_in_seed_sidecars": False,
            "model_answers_allowed_in_seed_sidecars": False,
            "seed_inputs_are_verifier_evidence": False,
            "stage_approves_command_bindings": False,
            "stage_executes_commands": False,
        },
        "config": {
            "sidecar_key": SIDECAR_KEY,
            "input_name": INPUT_NAME,
            "target_flag": TARGET_FLAG,
            "requires_sidecar_audit_ready": True,
        },
        "summary": summary,
        "paths": {
            "report": str(report_path),
            "command_bindings": str(command_bindings_path),
            "source_family_url_seeds": None if seed_sidecar_path is None else str(seed_sidecar_path),
            "seed_binding_records": str(records_path),
            "artifact_manifest": str(manifest_path),
        },
        "records": records,
        "updated_bindings": updated_bindings,
        "metadata": dict(metadata or {}),
    }

    _write_json(report_path, payload, compact=compact_json)
    _write_json(command_bindings_path, updated_bindings, compact=compact_json)
    _write_jsonl(records_path, records, compact=compact_json)
    manifest = _write_manifest(
        manifest_path=manifest_path,
        output_path=report_path,
        command_bindings_path=command_bindings_path,
        records_path=records_path,
        audit_path=audit_path,
        base_bindings_path=bindings_path,
        seed_sidecar_path=seed_sidecar_path,
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
                "seed_sidecar": None if seed_sidecar_path is None else str(seed_sidecar_path),
                "seed_binding_record_count": summary["seed_binding_record_count"],
                "ready_seed_input_count": summary["ready_seed_input_count"],
                "blocked_seed_input_count": summary["blocked_seed_input_count"],
                "applied_input_count": summary["applied_input_count"],
                "applied_placeholder_count": summary["applied_placeholder_count"],
                "manifest_summary": manifest.get("summary", {}),
                **dict(metadata or {}),
            },
        ).save_json()
    return payload


def _source_family_seed_sidecar_path(
    audit: Mapping[str, Any],
    *,
    source_root: Path | None,
) -> Path | None:
    raw = _mapping(audit.get("paths")).get(SIDECAR_KEY)
    if not _clean(raw):
        return None
    return _resolve_path(str(raw), source_root=source_root)


def _seed_binding_records(
    audit: Mapping[str, Any],
    *,
    seed_rows: Sequence[Mapping[str, Any]],
    seed_sidecar_path: Path | None,
) -> tuple[dict[str, Any], ...]:
    sidecar_status = _mapping(_mapping(audit.get("summary")).get("sidecar_status_counts")).get(SIDECAR_KEY)
    sidecar_status = _mapping(sidecar_status)
    ready_count = _int_or_zero(sidecar_status.get("ready"))
    blocked_count = _int_or_zero(sidecar_status.get("blocked"))
    audited_row_count = _int_or_zero(_mapping(_mapping(audit.get("summary")).get("sidecar_counts")).get(SIDECAR_KEY))
    sidecar_missing = seed_sidecar_path is None or not seed_sidecar_path.exists()
    global_reasons: list[str] = []
    if sidecar_missing:
        global_reasons.append("missing_source_family_url_seed_sidecar")
    if ready_count <= 0:
        global_reasons.append("no_ready_source_family_url_seed_rows")
    if blocked_count > 0:
        global_reasons.append("source_family_url_seed_sidecar_blocked")
    if seed_rows and audited_row_count and len(seed_rows) != audited_row_count:
        global_reasons.append("source_family_url_seed_sidecar_changed_since_audit")
    if any(row.get("not_verifier_evidence") is not True for row in seed_rows):
        global_reasons.append("seed_sidecar_row_not_marked_non_evidence")

    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in seed_rows:
        action_id = _clean(row.get("action_id"))
        input_name = _clean(row.get("input_name")) or INPUT_NAME
        grouped.setdefault((action_id, input_name), []).append(row)
    if not grouped:
        grouped[("", INPUT_NAME)] = []

    records = []
    for (action_id, input_name), rows in sorted(grouped.items()):
        skip_reasons = list(global_reasons)
        if not action_id:
            skip_reasons.append("missing_action_id")
        if input_name != INPUT_NAME:
            skip_reasons.append("unexpected_input_name")
        task_ids = tuple(
            dict.fromkeys(
                _clean(row.get("task_id") or row.get("collection_task_id"))
                for row in rows
                if _clean(row.get("task_id") or row.get("collection_task_id"))
            )
        )
        status = "ready" if not skip_reasons else "blocked"
        records.append({
            "schema_version": 1,
            "workflow": WORKFLOW,
            "action_id": action_id,
            "input_name": input_name,
            "target_flag": TARGET_FLAG,
            "sidecar_key": SIDECAR_KEY,
            "status": status,
            "skip_reasons": tuple(dict.fromkeys(skip_reasons)),
            "sidecar_path": "" if seed_sidecar_path is None else str(seed_sidecar_path),
            "seed_row_count": len(rows),
            "task_ids": task_ids,
            "audit_ready_row_count": ready_count,
            "audit_blocked_row_count": blocked_count,
            "audited_row_count": audited_row_count,
            "not_verifier_evidence": bool(rows) and all(row.get("not_verifier_evidence") is True for row in rows),
        })
    return tuple(records)


def _updated_bindings(
    base: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    updated = dict(base)
    inputs = dict(_mapping(base.get("inputs")))
    bindings = {str(key): dict(_mapping(value)) for key, value in _mapping(base.get("bindings")).items()}
    ready_records = tuple(record for record in records if record.get("status") == "ready")
    applied_inputs = 0
    applied_placeholders = 0
    missing_action_bindings = 0
    unapplied_records = 0
    for record in ready_records:
        action_id = str(record.get("action_id") or "")
        input_name = str(record.get("input_name") or "")
        sidecar_path = str(record.get("sidecar_path") or "")
        if not action_id or not input_name or not sidecar_path:
            unapplied_records += 1
            continue
        binding = bindings.get(action_id)
        if binding is None:
            missing_action_bindings += 1
            unapplied_records += 1
            continue
        input_value = _input_value(record)
        entry_inputs = dict(_mapping(binding.get("inputs")))
        inputs[input_name] = input_value
        entry_inputs[input_name] = input_value
        reviews = list(_mapping_sequence(binding.get("source_backed_input_reviews", ())))
        reviews.append(_source_backed_input_review(record))
        command_result = _bind_flag_placeholder(
            _string_tuple(binding.get("bound_commands", ())),
            TARGET_FLAG,
            sidecar_path,
        )
        applied_inputs += 1
        applied_placeholders += command_result["applied_placeholder_count"]
        binding["inputs"] = entry_inputs
        binding["source_backed_input_reviews"] = tuple(reviews)
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
        "unapplied_seed_input_count": unapplied_records,
    }


def _bind_flag_placeholder(
    commands: Sequence[str],
    flag: str,
    value: str,
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
            if token == "..." and index > 0 and tokens[index - 1] == flag:
                tokens[index] = value
                applied += 1
        updated_commands.append(shlex.join(tokens))
    return {"commands": tuple(updated_commands), "applied_placeholder_count": applied}


def _input_value(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "path": str(record.get("sidecar_path") or ""),
        "source_workflow": WORKFLOW,
        "audit_ready_row_count": _int_or_zero(record.get("audit_ready_row_count")),
        "seed_row_count": _int_or_zero(record.get("seed_row_count")),
        "review_status": "needs_command_binding_review",
        "not_verifier_evidence": record.get("not_verifier_evidence") is True,
    }


def _source_backed_input_review(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "input_name": str(record.get("input_name") or ""),
        "target_flag": str(record.get("target_flag") or ""),
        "sidecar_key": str(record.get("sidecar_key") or ""),
        "sidecar_path": str(record.get("sidecar_path") or ""),
        "seed_row_count": _int_or_zero(record.get("seed_row_count")),
        "audit_ready_row_count": _int_or_zero(record.get("audit_ready_row_count")),
        "review_status": "needs_command_binding_review",
        "not_verifier_evidence": record.get("not_verifier_evidence") is True,
    }


def _summary(
    *,
    records: Sequence[Mapping[str, Any]],
    apply_summary: Mapping[str, Any],
    seed_rows: Sequence[Mapping[str, Any]],
    seed_sidecar_path: Path | None,
) -> dict[str, Any]:
    status_counts = Counter(str(record.get("status") or "") for record in records)
    skip_counts: Counter[str] = Counter()
    for record in records:
        for reason in _string_tuple(record.get("skip_reasons", ())):
            skip_counts[reason] += 1
    return {
        "seed_sidecar_count": 1 if seed_sidecar_path is not None else 0,
        "seed_sidecar_exists": bool(seed_sidecar_path is not None and seed_sidecar_path.exists()),
        "seed_sidecar_row_count": len(seed_rows),
        "seed_binding_record_count": len(records),
        "ready_seed_input_count": status_counts.get("ready", 0),
        "blocked_seed_input_count": sum(
            count for status, count in status_counts.items() if status != "ready"
        ),
        "applied_input_count": _int_or_zero(apply_summary.get("applied_input_count")),
        "applied_placeholder_count": _int_or_zero(
            apply_summary.get("applied_placeholder_count")
        ),
        "missing_action_binding_count": _int_or_zero(
            apply_summary.get("missing_action_binding_count")
        ),
        "unapplied_seed_input_count": _int_or_zero(
            apply_summary.get("unapplied_seed_input_count")
        ),
        "record_status_counts": _sorted_counter(status_counts),
        "skip_reason_counts": _sorted_counter(skip_counts),
    }


def _status(summary: Mapping[str, Any]) -> str:
    if _int_or_zero(summary.get("seed_sidecar_count")) == 0:
        return "empty"
    if _int_or_zero(summary.get("blocked_seed_input_count")) > 0:
        return "needs_review"
    if _int_or_zero(summary.get("missing_action_binding_count")) > 0:
        return "needs_review"
    if _int_or_zero(summary.get("unapplied_seed_input_count")) > 0:
        return "needs_review"
    if _int_or_zero(summary.get("applied_input_count")) <= 0:
        return "needs_review"
    return "ready_for_binding_review"


def _write_manifest(
    *,
    manifest_path: Path,
    output_path: Path,
    command_bindings_path: Path,
    records_path: Path,
    audit_path: Path | None,
    base_bindings_path: Path | None,
    seed_sidecar_path: Path | None,
    payload: Mapping[str, Any],
    metadata: Mapping[str, Any],
    compact: bool,
) -> Mapping[str, Any]:
    manifest = build_artifact_manifest(
        {
            "source_family_seed_binding_staging_report": output_path,
            "frontier_research_queue_command_bindings": command_bindings_path,
            "seed_binding_records": records_path,
            "input_binding_audit": audit_path,
            "base_bindings": base_bindings_path,
            "source_family_url_seeds": seed_sidecar_path,
        },
        root=manifest_path.parent,
        metadata={
            "runner": "bind_frontier_research_queue_seed_inputs",
            "workflow": WORKFLOW,
            "status": payload.get("status"),
            "seed_sidecar_row_count": _nested(payload, "summary", "seed_sidecar_row_count"),
            "applied_input_count": _nested(payload, "summary", "applied_input_count"),
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


def _resolve_path(path: str, *, source_root: Path | None) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    if source_root is not None and (source_root / candidate).exists():
        return source_root / candidate
    if candidate.exists():
        return candidate
    return ROOT / candidate


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
    parser.add_argument("--input-binding-audit", required=True)
    parser.add_argument("--base-bindings", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--json", default=None)
    parser.add_argument("--bindings-json", default=None)
    parser.add_argument("--records-jsonl", default=None)
    parser.add_argument("--artifact-manifest", default=None)
    parser.add_argument("--registry", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--version", default=None)
    parser.add_argument("--compact-json", action="store_true")
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    payload = bind_frontier_research_queue_seed_inputs(
        input_binding_audit=args.input_binding_audit,
        base_bindings=args.base_bindings,
        output_dir=args.output_dir,
        json_path=args.json,
        bindings_json_path=args.bindings_json,
        records_jsonl_path=args.records_jsonl,
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
