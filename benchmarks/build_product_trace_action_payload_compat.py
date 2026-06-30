"""Build replay-compatible ProductTrace action-payload copies.

This utility is intentionally narrow: it preserves saved ProductTrace payloads
while repairing legacy ``retrieve`` actions that predate executable retrieval
targets. The output is a source artifact for action-gated replay workflows, not
new model, verifier, or retrieval evidence.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eigentruth.control.policy import ControlAction  # noqa: E402
from eigentruth.json_utils import strict_json_dumps  # noqa: E402
from eigentruth.registry import (  # noqa: E402
    ArtifactRegistry,
    build_artifact_manifest,
)

WORKFLOW = "product_trace_action_payload_compat"
DEFAULT_NAME = "smollm2-product-trace-action-payload-compat"
DEFAULT_VERSION = "0.1"


def run(
    *,
    trace_globs: Sequence[str],
    output_dir: str | Path,
    report_path: str | Path | None = None,
    artifact_manifest_path: str | Path | None = None,
    registry_path: str | Path | None = None,
    name: str = DEFAULT_NAME,
    version: str = DEFAULT_VERSION,
    metadata: Mapping[str, Any] | None = None,
    compact_json: bool = False,
) -> dict[str, Any]:
    """Copy traces and repair legacy empty retrieval-target payloads."""
    output_root = Path(output_dir)
    traces_root = output_root / "traces"
    resolved_report_path = Path(report_path) if report_path is not None else output_root / "action-payload-compat.json"
    resolved_manifest_path = (
        Path(artifact_manifest_path)
        if artifact_manifest_path is not None
        else output_root / "artifact-manifest.json"
    )
    trace_paths = _resolve_trace_paths(trace_globs)
    if not trace_paths:
        raise ValueError("at least one trace path must match --trace-glob.")

    written_paths = []
    modified_count = 0
    repaired_action_count = 0
    added_target_count = 0
    blocked_count = 0
    for trace_path in trace_paths:
        payload = _load_json(trace_path)
        repaired, action_count, target_count, blocked = _repair_trace_payload(payload, source_path=trace_path)
        relative_path = _relative_trace_path(trace_path)
        output_path = traces_root / relative_path
        _write_json(output_path, payload, compact=compact_json)
        written_paths.append(output_path)
        modified_count += int(repaired)
        repaired_action_count += action_count
        added_target_count += target_count
        blocked_count += blocked

    status = "ready" if blocked_count == 0 else "blocked"
    report = {
        "schema_version": 1,
        "workflow": WORKFLOW,
        "status": status,
        "summary": {
            "input_trace_count": len(trace_paths),
            "output_trace_count": len(written_paths),
            "modified_trace_count": modified_count,
            "repaired_action_count": repaired_action_count,
            "added_retrieval_target_count": added_target_count,
            "blocked_trace_count": blocked_count,
        },
        "artifact_manifest_summary": {
            "artifact_count": 2,
            "directory_count": 1,
            "file_count": 1,
            "missing_count": 0,
        },
        "label_usage": {
            "labels_used": False,
            "model_outputs_generated": False,
            "verifier_outputs_generated": False,
            "retrieval_evidence_generated": False,
            "legacy_action_payloads_repaired": True,
        },
        "paths": {
            "report": str(resolved_report_path),
            "artifact_manifest": str(resolved_manifest_path),
            "traces_dir": str(traces_root),
        },
        "metadata": dict(metadata or {}),
    }
    _write_json(resolved_report_path, report, compact=compact_json)
    manifest = build_artifact_manifest(
        {
            "report": resolved_report_path,
            "traces": traces_root,
        },
        root=resolved_manifest_path.parent,
        metadata={
            "workflow": WORKFLOW,
            "status": status,
            "input_trace_count": len(trace_paths),
            "modified_trace_count": modified_count,
            "repaired_action_count": repaired_action_count,
            **dict(metadata or {}),
        },
    )
    _write_json(resolved_manifest_path, manifest, compact=compact_json)

    if registry_path is not None:
        ArtifactRegistry.load_json(registry_path).record_report(
            name=name,
            version=version,
            path=resolved_report_path,
            metadata={
                "workflow": WORKFLOW,
                "status": status,
                "artifact_manifest": str(resolved_manifest_path),
                "input_trace_count": len(trace_paths),
                "modified_trace_count": modified_count,
                "repaired_action_count": repaired_action_count,
                **dict(metadata or {}),
            },
        ).save_json(registry_path)
    return report


def _repair_trace_payload(payload: dict[str, Any], *, source_path: Path) -> tuple[bool, int, int, int]:
    actions = payload.get("actions")
    if not isinstance(actions, list):
        return False, 0, 0, 0
    claims = _claim_targets(payload.get("claims"))
    repaired = False
    repaired_action_count = 0
    added_target_count = 0
    blocked_count = 0
    for action in actions:
        if not isinstance(action, dict):
            continue
        if str(action.get("action", "")).strip() != ControlAction.RETRIEVE.value:
            continue
        body = action.get("payload")
        if not isinstance(body, dict):
            continue
        if _executable_retrieval_query_count(body):
            continue
        if not claims:
            blocked_count += 1
            continue
        body["retrieval_targets"] = tuple(claims)
        compat = dict(body.get("compatibility", {})) if isinstance(body.get("compatibility"), Mapping) else {}
        compat["action_payload_compat"] = {
            "workflow": WORKFLOW,
            "source_path": str(source_path),
            "repair": "filled_empty_retrieval_targets_from_claims",
            "target_count": len(claims),
        }
        body["compatibility"] = compat
        repaired = True
        repaired_action_count += 1
        added_target_count += len(claims)
    if repaired:
        metadata = payload.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
            payload["metadata"] = metadata
        compat_meta = (
            dict(metadata.get("compatibility", {}))
            if isinstance(metadata.get("compatibility"), Mapping)
            else {}
        )
        compat_meta[WORKFLOW] = {
            "source_path": str(source_path),
            "repaired_action_count": repaired_action_count,
            "added_retrieval_target_count": added_target_count,
        }
        metadata["compatibility"] = compat_meta
    return repaired, repaired_action_count, added_target_count, blocked_count


def _claim_targets(value: Any) -> tuple[dict[str, Any], ...]:
    targets = []
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return ()
    for index, claim in enumerate(value, start=1):
        if not isinstance(claim, Mapping):
            continue
        text = str(claim.get("text", "")).strip()
        if not text:
            continue
        raw_claim_id = claim.get("claim_id")
        claim_id = str(raw_claim_id).strip() if raw_claim_id is not None else f"c{index}"
        target = {
            "claim_id": claim_id,
            "text": text,
        }
        metadata = claim.get("metadata")
        if isinstance(metadata, Mapping):
            target["metadata"] = dict(metadata)
        targets.append(target)
    return tuple(targets)


def _executable_retrieval_query_count(payload: Mapping[str, Any]) -> int:
    count = 0
    query = payload.get("query")
    if isinstance(query, str) and query.strip():
        count += 1
    targets = payload.get("retrieval_targets", ())
    if isinstance(targets, Mapping):
        targets = (targets,)
    if not isinstance(targets, Sequence) or isinstance(targets, (str, bytes, bytearray)):
        targets = (targets,)
    for target in targets:
        if isinstance(target, str) and target.strip():
            count += 1
        elif isinstance(target, Mapping):
            text = target.get("text", target.get("query", target.get("claim_text")))
            if isinstance(text, str) and text.strip():
                count += 1
    return count


def _relative_trace_path(path: Path) -> Path:
    parts = path.parts
    if "traces" in parts:
        index = len(parts) - 1 - tuple(reversed(parts)).index("traces")
        return Path(*parts[index + 1 :])
    return Path(path.name)


def _resolve_trace_paths(patterns: Sequence[str]) -> tuple[Path, ...]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern, recursive=True))
        paths.extend(Path(match) for match in matches if Path(match).is_file())
    return tuple(dict.fromkeys(paths))


def _load_json(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"trace payload must be a JSON object: {path}")
    return loaded


def _write_json(path: str | Path, payload: Mapping[str, Any], *, compact: bool) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if compact:
        text = strict_json_dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    else:
        text = strict_json_dumps(payload, indent=2, sort_keys=True) + "\n"
    output.write_text(text, encoding="utf-8")


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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-glob", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--json", default=None)
    parser.add_argument("--artifact-manifest", default=None)
    parser.add_argument("--registry", default=None)
    parser.add_argument("--name", default=DEFAULT_NAME)
    parser.add_argument("--version", default=DEFAULT_VERSION)
    parser.add_argument("--metadata", action="append", default=[])
    parser.add_argument("--compact-json", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = run(
        trace_globs=tuple(args.trace_glob),
        output_dir=args.output_dir,
        report_path=args.json,
        artifact_manifest_path=args.artifact_manifest,
        registry_path=args.registry,
        name=args.name,
        version=args.version,
        metadata=_parse_metadata(args.metadata),
        compact_json=bool(args.compact_json),
    )
    print(strict_json_dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
