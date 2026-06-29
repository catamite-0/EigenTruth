"""Build rerun queue items for blocked frontier detectability evidence."""

from __future__ import annotations

import argparse
import json
import sys
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

WORKFLOW = "frontier_detectability_evidence_rerun_queue"


def build_frontier_detectability_evidence_rerun_queue(
    *,
    source: str | Path,
    json_path: str | Path | None = None,
    artifact_manifest_path: str | Path | None = None,
    registry_path: str | Path | None = None,
    name: str | None = None,
    version: str | None = None,
    output_dir: str | Path | None = None,
    score_paths: Sequence[str | Path] = (),
    consistency_signal: str | None = None,
    confidence_signal: str | None = None,
    consistency_direction: str = "higher",
    confidence_direction: str = "higher",
    cell: str = "entrenched",
    false_only: bool = True,
    max_records: int = 100,
    python_executable: str = sys.executable,
) -> dict[str, Any]:
    """Load frontier/gap/detectability evidence and build actionable queue rows."""
    if artifact_manifest_path is not None and json_path is None:
        raise ValueError("artifact_manifest_path requires json_path.")
    if registry_path is not None and (not name or not version):
        raise ValueError("registry_path requires name and version.")
    if consistency_direction not in {"higher", "lower"}:
        raise ValueError("consistency_direction must be one of: higher, lower.")
    if confidence_direction not in {"higher", "lower"}:
        raise ValueError("confidence_direction must be one of: higher, lower.")
    if int(max_records) < 0:
        raise ValueError("max_records must be >= 0.")
    source_path = Path(source)
    output_path = None if json_path is None else Path(json_path)
    manifest_path = None if artifact_manifest_path is None else Path(artifact_manifest_path)
    rerun_root = Path(output_dir) if output_dir is not None else source_path.parent / "frontier-detectability-reruns"
    payload = _load_json_object(source_path)
    source_dir = source_path.parent
    default_payload = _source_defaults_payload(payload, source_dir=source_dir)
    reports = _load_detectability_reports(payload, source_path=source_path, source_dir=source_dir)
    if payload.get("workflow") == "evidence_gap_plan":
        reports.update(_load_detectability_reports(default_payload, source_path=source_path, source_dir=source_dir))
    blocked_run_names = _blocked_detectability_run_names(payload) or _blocked_detectability_run_names(default_payload)
    if payload.get("workflow") == "detectability_taxonomy" and not blocked_run_names:
        blocked_run_names = (_detectability_report_name(source_path, payload),)

    entries = _entries_from_reports(
        reports,
        blocked_run_names=blocked_run_names,
        rerun_root=rerun_root,
        cell=cell,
        false_only=false_only,
        max_records=max_records,
        python_executable=python_executable,
    )
    if not entries and _detectability_track_requested(payload, default_payload):
        entries = _entries_from_score_paths(
            score_paths,
            rerun_root=rerun_root,
            consistency_signal=consistency_signal,
            confidence_signal=confidence_signal,
            consistency_direction=consistency_direction,
            confidence_direction=confidence_direction,
            python_executable=python_executable,
        )
        if not entries:
            entries = (_missing_inputs_entry(rerun_root=rerun_root, score_paths=score_paths),)

    command_count = sum(1 for entry in entries if entry["command_status"] == "ready")
    output = {
        "schema_version": 1,
        "workflow": WORKFLOW,
        "status": "ready" if entries else "empty",
        "source": str(source_path),
        "summary": {
            "blocked_run_count": len(entries),
            "blind_spot_analysis_count": sum(
                1 for entry in entries if entry["command_kind"] == "blind_spot_analysis"
            ),
            "taxonomy_rerun_count": sum(1 for entry in entries if entry["command_kind"] == "taxonomy_report"),
            "command_count": command_count,
            "missing_command_count": len(entries) - command_count,
        },
        "paths": {
            "rerun_queue": None if output_path is None else str(output_path),
            "artifact_manifest": None if manifest_path is None else str(manifest_path),
        },
        "config": {
            "scores": tuple(str(path) for path in score_paths),
            "consistency_signal": consistency_signal,
            "confidence_signal": confidence_signal,
            "consistency_direction": consistency_direction,
            "confidence_direction": confidence_direction,
            "cell": cell,
            "false_only": bool(false_only),
            "max_records": int(max_records),
            "python_executable": python_executable,
        },
        "entries": entries,
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(strict_json_dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = None
    if manifest_path is not None:
        manifest = _write_artifact_manifest(
            source_path=source_path,
            output_path=output_path,
            manifest_path=manifest_path,
            payload=output,
        )
    if registry_path is not None:
        assert name is not None and version is not None
        _record_registry(
            registry_path=Path(registry_path),
            name=name,
            version=version,
            report_path=output_path if output_path is not None else source_path,
            manifest_path=manifest_path,
            payload=output,
            manifest=manifest,
        )
    return output


def run(args: argparse.Namespace) -> dict[str, Any]:
    payload = build_frontier_detectability_evidence_rerun_queue(
        source=args.source,
        json_path=args.json,
        artifact_manifest_path=args.artifact_manifest,
        registry_path=args.registry,
        name=args.name,
        version=args.version,
        output_dir=args.output_dir,
        score_paths=tuple(args.scores or ()),
        consistency_signal=args.consistency_signal,
        confidence_signal=args.confidence_signal,
        consistency_direction=args.consistency_direction,
        confidence_direction=args.confidence_direction,
        cell=args.cell,
        false_only=not args.include_true,
        max_records=int(args.max_records),
        python_executable=args.python,
    )
    summary = payload["summary"]
    print(
        "frontier_detectability_evidence_rerun_queue="
        f"{payload['status']} "
        f"blocked_runs={summary['blocked_run_count']} "
        f"commands={summary['command_count']}"
    )
    return payload


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="frontier release, evidence-gap, or taxonomy JSON")
    parser.add_argument("--json", default=None, help="optional output JSON path")
    parser.add_argument("--artifact-manifest", default=None, help="optional artifact manifest JSON path")
    parser.add_argument("--registry", default=None, help="optional local ArtifactRegistry JSON path")
    parser.add_argument("--name", default=None, help="registry record name")
    parser.add_argument("--version", default=None, help="registry record version")
    parser.add_argument("--output-dir", default=None, help="root directory for rerun outputs")
    parser.add_argument("--scores", action="append", default=[], help="name=score_dump path; repeatable")
    parser.add_argument("--consistency-signal", default=None)
    parser.add_argument("--confidence-signal", default=None)
    parser.add_argument("--consistency-direction", choices=("higher", "lower"), default="higher")
    parser.add_argument("--confidence-direction", choices=("higher", "lower"), default="higher")
    parser.add_argument("--cell", default="entrenched")
    parser.add_argument("--include-true", action="store_true")
    parser.add_argument("--max-records", type=int, default=100)
    parser.add_argument("--python", default=sys.executable, help="Python executable for generated commands")
    run(parser.parse_args(argv))


def _entries_from_reports(
    reports: Mapping[str, tuple[Path, Mapping[str, Any]]],
    *,
    blocked_run_names: Sequence[str],
    rerun_root: Path,
    cell: str,
    false_only: bool,
    max_records: int,
    python_executable: str,
) -> tuple[dict[str, Any], ...]:
    selected_names = tuple(blocked_run_names) or tuple(sorted(reports))
    entries = []
    for run_name in selected_names:
        report_item = reports.get(run_name)
        if report_item is None:
            continue
        report_path, report = report_item
        output_dir = rerun_root / _slug(run_name)
        command = [
            python_executable,
            "benchmarks/analyze_detectability_blind_spots.py",
            "--taxonomy-report",
            str(report_path),
            "--cell",
            cell,
            "--max-records",
            str(int(max_records)),
            "--json",
            str(output_dir / "blind-spots.json"),
            "--artifact-manifest",
            str(output_dir / "artifact-manifest.json"),
        ]
        score_dump_path = _nested(report, "source", "score_dump_path")
        _append_optional_value(command, "--scores", score_dump_path)
        if not false_only:
            command.append("--include-true")
        entries.append({
            "run": run_name,
            "command_kind": "blind_spot_analysis",
            "source_report": str(report_path),
            "source_score_dump": score_dump_path,
            "rerun_output_dir": str(output_dir),
            "command_status": "ready",
            "missing_inputs": (),
            "command": tuple(command),
            "dry_run_command": None,
        })
    return tuple(entries)


def _entries_from_score_paths(
    score_paths: Sequence[str | Path],
    *,
    rerun_root: Path,
    consistency_signal: str | None,
    confidence_signal: str | None,
    consistency_direction: str,
    confidence_direction: str,
    python_executable: str,
) -> tuple[dict[str, Any], ...]:
    if not score_paths or not consistency_signal or not confidence_signal:
        return ()
    entries = []
    for raw_score_path in score_paths:
        run_name, score_path = _parse_named_path(str(raw_score_path))
        output_dir = rerun_root / _slug(run_name)
        command = (
            python_executable,
            "benchmarks/eval_detectability_taxonomy.py",
            "--scores",
            score_path,
            "--consistency-signal",
            consistency_signal,
            "--confidence-signal",
            confidence_signal,
            "--consistency-direction",
            consistency_direction,
            "--confidence-direction",
            confidence_direction,
            "--metadata",
            f"run_name={run_name}",
            "--json",
            str(output_dir / "detectability-taxonomy-report.json"),
        )
        entries.append({
            "run": run_name,
            "command_kind": "taxonomy_report",
            "source_report": None,
            "source_score_dump": score_path,
            "rerun_output_dir": str(output_dir),
            "command_status": "ready",
            "missing_inputs": (),
            "command": command,
            "dry_run_command": None,
        })
    return tuple(entries)


def _missing_inputs_entry(*, rerun_root: Path, score_paths: Sequence[str | Path]) -> dict[str, Any]:
    missing = []
    if not score_paths:
        missing.append("scores")
    missing.extend(("consistency_signal", "confidence_signal"))
    return {
        "run": None,
        "command_kind": None,
        "source_report": None,
        "source_score_dump": None,
        "rerun_output_dir": str(rerun_root),
        "command_status": "missing_inputs",
        "missing_inputs": tuple(dict.fromkeys(missing)),
        "command": None,
        "dry_run_command": None,
    }


def _blocked_detectability_run_names(payload: Mapping[str, Any]) -> tuple[str, ...]:
    workflow = payload.get("workflow")
    names = []
    if workflow == "frontier_release_evidence_comparison":
        for decision in _mapping_sequence(payload.get("run_decisions", ())):
            detectability = _mapping(decision.get("detectability_decision"))
            if detectability.get("status") == "blocked":
                names.append(_optional_str(decision.get("name")) or _optional_str(detectability.get("name")))
    elif workflow == "evidence_gap_plan":
        for gap in _mapping_sequence(payload.get("gaps", ())):
            metadata = _mapping(gap.get("metadata"))
            evidence_kind = _optional_str(metadata.get("evidence_kind"))
            gate = _optional_str(gap.get("gate"))
            if evidence_kind == "detectability_taxonomy" or gate == "detectability_taxonomy":
                names.extend(_string_tuple(metadata.get("detectability_blocked_runs")))
    return _unique_strings(name for name in names if name)


def _detectability_track_requested(payload: Mapping[str, Any], default_payload: Mapping[str, Any]) -> bool:
    for candidate in (payload, default_payload):
        workflow = candidate.get("workflow")
        if workflow == "detectability_taxonomy":
            return True
        if workflow == "frontier_release_evidence_comparison":
            decision = _mapping(candidate.get("decision"))
            if decision.get("detectability_track_status") == "blocked":
                return True
        if workflow == "evidence_gap_plan":
            for gap in _mapping_sequence(candidate.get("gaps", ())):
                metadata = _mapping(gap.get("metadata"))
                if metadata.get("evidence_kind") == "detectability_taxonomy":
                    return True
                if gap.get("gate") == "detectability_taxonomy":
                    return True
    return False


def _load_detectability_reports(
    payload: Mapping[str, Any],
    *,
    source_path: Path,
    source_dir: Path,
) -> dict[str, tuple[Path, Mapping[str, Any]]]:
    reports: dict[str, tuple[Path, Mapping[str, Any]]] = {}
    if payload.get("workflow") == "detectability_taxonomy":
        reports[_detectability_report_name(source_path, payload)] = (source_path, payload)
    inputs = _mapping(payload.get("inputs"))
    for item in _mapping_sequence(inputs.get("detectability_taxonomy_reports", ())):
        raw_path = item.get("path")
        if raw_path is None:
            continue
        path = _resolve_report_path(Path(str(raw_path)), source_dir=source_dir)
        if path is None:
            continue
        report = _load_json_object(path)
        if report.get("workflow") == "detectability_taxonomy":
            reports[_detectability_report_name(path, report)] = (path, report)
    return reports


def _source_defaults_payload(payload: Mapping[str, Any], *, source_dir: Path) -> Mapping[str, Any]:
    if payload.get("workflow") != "evidence_gap_plan":
        return payload
    source = _optional_str(payload.get("source_path"))
    if not source:
        return payload
    resolved = _resolve_report_path(Path(source), source_dir=source_dir)
    if resolved is None:
        return payload
    return _load_json_object(resolved)


def _detectability_report_name(path: Path, report: Mapping[str, Any]) -> str:
    metadata = _mapping(report.get("metadata"))
    for key in ("run_name", "name"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    source = _mapping(report.get("source"))
    summary = _mapping(source.get("score_dump_summary"))
    for key in ("name", "run_name", "model"):
        value = summary.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return path.stem


def _parse_named_path(value: str) -> tuple[str, str]:
    if "=" not in value:
        path = Path(value)
        return path.stem, str(path)
    name, path = value.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError("score path name cannot be empty.")
    return name, path


def _write_artifact_manifest(
    *,
    source_path: Path,
    output_path: Path | None,
    manifest_path: Path,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    summary = _mapping(payload.get("summary"))
    manifest = build_artifact_manifest(
        {
            "source": source_path,
            "frontier_detectability_evidence_rerun_queue": output_path,
        },
        root=manifest_path.parent,
        metadata={
            "runner": "plan_frontier_detectability_evidence_reruns",
            "status": payload.get("status"),
            "source": str(source_path),
            "blocked_run_count": summary.get("blocked_run_count"),
            "command_count": summary.get("command_count"),
            "missing_command_count": summary.get("missing_command_count"),
        },
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(strict_json_dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _record_registry(
    *,
    registry_path: Path,
    name: str,
    version: str,
    report_path: Path,
    manifest_path: Path | None,
    payload: Mapping[str, Any],
    manifest: Mapping[str, Any] | None,
) -> None:
    summary = _mapping(payload.get("summary"))
    ArtifactRegistry.load_json(registry_path).record_report(
        name=name,
        version=version,
        path=report_path,
        metadata={
            "workflow": WORKFLOW,
            "status": payload.get("status"),
            "source": payload.get("source"),
            "blocked_run_count": summary.get("blocked_run_count"),
            "command_count": summary.get("command_count"),
            "missing_command_count": summary.get("missing_command_count"),
            "artifact_manifest": None if manifest_path is None else str(manifest_path),
            "manifest_summary": None if manifest is None else _mapping(manifest.get("summary")),
        },
    ).save_json()


def _load_json_object(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"source JSON must contain an object: {path}")
    return data


def _resolve_report_path(path: Path, *, source_dir: Path) -> Path | None:
    if path.is_absolute():
        return path if path.exists() else None
    candidates = (source_dir / path, ROOT / path, Path.cwd() / path)
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            return resolved
    return None


def _append_optional_value(command: list[str], flag: str, value: Any) -> None:
    if value is None:
        return
    text = str(value).strip()
    if text:
        command.extend((flag, text))


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _mapping_sequence(value: Any) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return ()
    return tuple(item for item in value if isinstance(item, Mapping))


def _string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,) if value.strip() else ()
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return tuple(str(item) for item in value if str(item).strip())
    return (str(value),)


def _unique_strings(values: Sequence[Any]) -> tuple[str, ...]:
    seen: set[str] = set()
    unique = []
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        unique.append(text)
    return tuple(unique)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _nested(payload: Mapping[str, Any], *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _slug(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in value).strip("-") or "run"


if __name__ == "__main__":
    main()
