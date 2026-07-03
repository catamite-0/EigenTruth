"""Build a rerun queue for blocked frontier multiple-testing cells."""

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


def build_frontier_multiple_testing_rerun_queue(
    *,
    source: str | Path,
    json_path: str | Path | None = None,
    artifact_manifest_path: str | Path | None = None,
    registry_path: str | Path | None = None,
    name: str | None = None,
    version: str | None = None,
    output_dir: str | Path | None = None,
    python_executable: str = sys.executable,
) -> dict[str, Any]:
    """Load release/gap evidence and build one queue item per blocked frontier cell."""
    if artifact_manifest_path is not None and json_path is None:
        raise ValueError("artifact_manifest_path requires json_path.")
    if registry_path is not None and (not name or not version):
        raise ValueError("registry_path requires name and version.")
    source_path = Path(source)
    output_path = None if json_path is None else Path(json_path)
    manifest_path = None if artifact_manifest_path is None else Path(artifact_manifest_path)
    payload = _load_json_object(source_path)
    source_dir = source_path.parent
    rerun_root = Path(output_dir) if output_dir is not None else source_dir / "frontier-multiple-testing-reruns"
    workflow_reports = _load_frontier_workflow_reports(payload, source_path=source_path, source_dir=source_dir)
    blocked_cells = _unique_cells((
        *_blocked_cells_from_payload(payload),
        *_cells_from_missing_workflow_gates(payload, workflow_reports=workflow_reports),
    ))
    entries = []
    for cell in blocked_cells:
        entry = _queue_entry(
            cell,
            workflow_reports=workflow_reports,
            rerun_root=rerun_root,
            python_executable=python_executable,
        )
        entries.append(entry)
    entries = _unique_entries(entries)
    command_count = sum(1 for entry in entries if entry.get("command"))
    output = {
        "schema_version": 1,
        "workflow": "frontier_multiple_testing_rerun_queue",
        "status": "ready" if entries else "empty",
        "source": str(source_path),
        "summary": {
            "blocked_cell_count": len(entries),
            "command_count": command_count,
            "missing_command_count": len(entries) - command_count,
        },
        "paths": {
            "rerun_queue": None if output_path is None else str(output_path),
            "artifact_manifest": None if manifest_path is None else str(manifest_path),
        },
        "entries": tuple(entries),
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
        _record_registry(
            registry_path=Path(registry_path),
            name=str(name),
            version=str(version),
            report_path=output_path if output_path is not None else source_path,
            manifest_path=manifest_path,
            payload=output,
            manifest=manifest,
        )
    return output


def run(args: argparse.Namespace) -> dict[str, Any]:
    payload = build_frontier_multiple_testing_rerun_queue(
        source=args.source,
        json_path=args.json,
        artifact_manifest_path=args.artifact_manifest,
        registry_path=args.registry,
        name=args.name,
        version=args.version,
        output_dir=args.output_dir,
        python_executable=args.python,
    )
    summary = payload["summary"]
    print(
        "frontier_multiple_testing_rerun_queue="
        f"{payload['status']} "
        f"blocked_cells={summary['blocked_cell_count']} "
        f"commands={summary['command_count']}"
    )
    return payload


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Build executable rerun queue items for blocked frontier multiple-testing cells"
    )
    parser.add_argument("--source", required=True, help="frontier release, workflow, or evidence-gap JSON")
    parser.add_argument("--json", default=None, help="optional output JSON path")
    parser.add_argument("--artifact-manifest", default=None, help="optional artifact manifest JSON path")
    parser.add_argument("--registry", default=None, help="optional local ArtifactRegistry JSON path")
    parser.add_argument("--name", default=None, help="registry record name")
    parser.add_argument("--version", default=None, help="registry record version")
    parser.add_argument("--output-dir", default=None, help="root directory for per-cell rerun outputs")
    parser.add_argument("--python", default=sys.executable, help="Python executable for generated commands")
    run(parser.parse_args(argv))


def _blocked_cells_from_payload(payload: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    workflow = payload.get("workflow")
    cells: list[dict[str, Any]] = []
    if workflow == "frontier_release_evidence_comparison":
        evidence_summary = _mapping(payload.get("evidence_summary"))
        cells.extend(_cells_from_metadata(evidence_summary, source_workflow=str(workflow)))
        for decision in _mapping_sequence(payload.get("multiple_testing_decisions", ())):
            metrics = _mapping(decision.get("metrics"))
            run_name = _optional_str(decision.get("name"))
            cells.extend(_cells_from_metadata(metrics, source_workflow=str(workflow), run_name=run_name))
            cells.extend(_cells_from_missing_gate_decision(decision, source_workflow=str(workflow)))
    elif workflow == "evidence_gap_plan":
        for gap in _mapping_sequence(payload.get("gaps", ())):
            metadata = _mapping(gap.get("metadata"))
            cells.extend(_cells_from_metadata(metadata, source_workflow=str(workflow)))
    elif workflow == "truthfulqa_frontier_workflow":
        cells.extend(_cells_from_frontier_workflow(payload, source_workflow=str(workflow)))
    return tuple(_unique_cells(cells))


def _cells_from_metadata(
    metadata: Mapping[str, Any],
    *,
    source_workflow: str,
    run_name: str | None = None,
) -> tuple[dict[str, Any], ...]:
    cells = []
    for key in ("multiple_testing_failed_cells", "multiple_testing_unknown_cells", "multiple_testing_blocked_cells"):
        for item in _mapping_sequence(metadata.get(key, ())):
            cells.append(_cell_record(item, source_workflow=source_workflow, run_name=run_name))
    return tuple(cells)


def _cells_from_missing_gate_decision(
    decision: Mapping[str, Any],
    *,
    source_workflow: str,
) -> tuple[dict[str, Any], ...]:
    metrics = _mapping(decision.get("metrics"))
    if metrics.get("gate_present") is True:
        return ()
    if _optional_str(metrics.get("workflow")) != "truthfulqa_frontier_workflow":
        return ()
    return _cells_from_frontier_workflow(
        metrics,
        source_workflow=source_workflow,
        run_name=_optional_str(decision.get("name")),
    )


def _cells_from_frontier_workflow(
    payload: Mapping[str, Any],
    *,
    source_workflow: str,
    run_name: str | None = None,
) -> tuple[dict[str, Any], ...]:
    cells: list[dict[str, Any]] = []
    gate = payload.get("multiple_testing_gate")
    if isinstance(gate, Mapping) and gate:
        for item in _mapping_sequence(gate.get("cells", ())):
            passed = item.get("pass")
            if passed is True:
                continue
            cells.append(_cell_record(item, source_workflow=source_workflow, run_name=run_name))
        return tuple(cells)
    config = _mapping(payload.get("config"))
    models = _mapping_sequence(config.get("models", ()))
    scales = _mapping_sequence(config.get("scales", ()))
    if not models or not scales:
        return ()
    for model in models:
        model_name = _optional_str(model.get("name"))
        if model_name is None:
            continue
        for scale in scales:
            scale_name = _optional_str(scale.get("name"))
            if scale_name is None:
                continue
            cells.append({
                "run": run_name,
                "cell": f"{model_name}-{scale_name}",
                "status": "missing_gate",
                "false_alarm": None,
                "detection": None,
                "report": None,
                "calibration": None,
                "source_workflow": source_workflow,
            })
    return tuple(cells)


def _cells_from_missing_workflow_gates(
    payload: Mapping[str, Any],
    *,
    workflow_reports: Sequence[tuple[Path, Mapping[str, Any]]],
) -> tuple[dict[str, Any], ...]:
    if payload.get("workflow") == "truthfulqa_frontier_workflow":
        return ()
    if not _has_missing_multiple_testing_gate(payload):
        return ()
    source_workflow = str(payload.get("workflow") or "unknown")
    run_names = _missing_gate_run_names(payload)
    cells: list[dict[str, Any]] = []
    for path, report in workflow_reports:
        report_name = _workflow_report_name(path, report)
        if run_names and report_name not in run_names and path.stem not in run_names:
            continue
        gate = report.get("multiple_testing_gate")
        if isinstance(gate, Mapping) and gate:
            continue
        cells.extend(
            _cells_from_frontier_workflow(
                report,
                source_workflow=source_workflow,
                run_name=report_name,
            )
        )
    return tuple(cells)


def _has_missing_multiple_testing_gate(payload: Mapping[str, Any]) -> bool:
    if payload.get("workflow") == "truthfulqa_frontier_workflow":
        return not bool(_mapping(payload.get("multiple_testing_gate")))
    for decision in _mapping_sequence(payload.get("multiple_testing_decisions", ())):
        if _decision_has_missing_gate_reason(decision):
            return True
    decision = _mapping(payload.get("decision"))
    return any(
        "multiple_testing_gate missing" in reason
        for reason in _string_tuple(decision.get("blocking_reasons", ()))
    )


def _missing_gate_run_names(payload: Mapping[str, Any]) -> tuple[str, ...]:
    names = []
    for decision in _mapping_sequence(payload.get("multiple_testing_decisions", ())):
        if _decision_has_missing_gate_reason(decision):
            name = _optional_str(decision.get("name"))
            if name:
                names.append(name)
    return tuple(dict.fromkeys(names))


def _decision_has_missing_gate_reason(decision: Mapping[str, Any]) -> bool:
    return any(
        "multiple_testing_gate missing" in reason
        for reason in _string_tuple(decision.get("blocking_reasons", ()))
    )


def _cell_record(
    item: Mapping[str, Any],
    *,
    source_workflow: str,
    run_name: str | None = None,
) -> dict[str, Any]:
    return {
        "run": _optional_str(item.get("run")) or run_name,
        "cell": _optional_str(item.get("cell")) or "",
        "status": _optional_str(item.get("status")) or _status_from_pass(item.get("pass")),
        "false_alarm": item.get("false_alarm"),
        "detection": item.get("detection"),
        "report": item.get("report"),
        "calibration": item.get("calibration"),
        "source_workflow": source_workflow,
    }


def _queue_entry(
    cell: Mapping[str, Any],
    *,
    workflow_reports: Sequence[tuple[Path, Mapping[str, Any]]],
    rerun_root: Path,
    python_executable: str,
) -> dict[str, Any]:
    cell_name = str(cell.get("cell") or "")
    workflow_path, workflow_report = _matching_workflow_report(cell, workflow_reports)
    command = None
    dry_run_command = None
    command_status = "missing_workflow_context"
    if workflow_report is not None:
        command = _rerun_command(
            workflow_report,
            cell_name=cell_name,
            output_dir=rerun_root / cell_name,
            python_executable=python_executable,
        )
        if command is not None:
            dry_run_command = tuple(command) + ("--dry-run",)
            command_status = "ready"
        else:
            command_status = "missing_cell_config"
    return {
        "run": cell.get("run"),
        "cell": cell_name,
        "status": cell.get("status"),
        "false_alarm": cell.get("false_alarm"),
        "detection": cell.get("detection"),
        "source_report": cell.get("report"),
        "source_calibration": cell.get("calibration"),
        "workflow_report": None if workflow_path is None else str(workflow_path),
        "rerun_output_dir": str(rerun_root / cell_name),
        "command_status": command_status,
        "command": command,
        "dry_run_command": dry_run_command,
    }


def _rerun_command(
    workflow_report: Mapping[str, Any],
    *,
    cell_name: str,
    output_dir: Path,
    python_executable: str,
) -> tuple[str, ...] | None:
    config = _mapping(workflow_report.get("config"))
    model, scale = _model_scale_for_cell(config, cell_name)
    if model is None or scale is None:
        return None
    multiple_testing_signals = _multiple_testing_signals_for_command(config)
    if not multiple_testing_signals:
        return None
    command: list[str] = [
        str(python_executable),
        "benchmarks/run_truthfulqa_frontier_workflow.py",
        "--output-dir",
        str(output_dir),
        "--model",
        f"{model['name']}={model['model_id']}",
        "--scale",
        _format_scale(scale),
        "--dtype",
        str(config.get("dtype") or "float32"),
        "--batch-size",
        str(config.get("batch_size") or 4),
        "--max-batch-tokens",
        str(config.get("max_batch_tokens") or 0),
        "--max-length",
        str(config.get("max_length") or 96),
        "--hidden-state-capture",
        str(config.get("hidden_state_capture") or "hooks"),
        "--covariance-mode",
        str(config.get("covariance_mode") or "full"),
        "--covariance-low-rank",
        str(config.get("covariance_low_rank") or 16),
        "--progress-every",
        str(config.get("progress_every") or 0),
        "--warmup-checkpoint-every",
        str(config.get("warmup_checkpoint_every") or 50),
        "--eval-reps-cache-shard-size",
        str(config.get("eval_reps_cache_shard_size") or 16),
        "--eval-reps-shard-read-cache-size",
        str(config.get("eval_reps_shard_read_cache_size") or 2),
        "--dump-scores-format",
        str(config.get("dump_scores_format") or "jsonl"),
        "--signals",
        _csv(config.get("signals", ())),
        "--conformal-signal",
        str(config.get("conformal_signal") or "maha_last"),
        "--conformal-repeats",
        str(config.get("conformal_repeats") or 20),
        "--ensemble-repeats",
        str(config.get("ensemble_repeats") or 20),
        "--seed",
        str(config.get("seed") or 0),
        "--artifact-alpha",
        str(config.get("artifact_alpha") or 0.10),
        "--multiple-testing-signals",
        multiple_testing_signals,
        "--multiple-testing-alpha",
        str(config.get("multiple_testing_alpha") or config.get("artifact_alpha") or 0.10),
        "--multiple-testing-method",
        str(config.get("multiple_testing_method") or "by"),
        "--best-alpha",
        str(config.get("best_alpha") or 0.10),
        "--best-by",
        str(config.get("best_by") or "auroc"),
        "--methods",
        _csv(config.get("ensemble_methods", ())),
        "--alphas",
        _csv(config.get("alphas", ())),
    ]
    score_dump = _score_dump_for_cell(workflow_report, cell_name)
    if score_dump is not None:
        command.extend(("--scores", score_dump))
    _append_optional_value(command, "--cache-dir", config.get("cache_dir"))
    _append_optional_value(command, "--attn-implementation", config.get("attn_implementation"))
    _append_optional_value(command, "--sweep-layers-from-band-report", config.get("sweep_layers_from_band_report"))
    _append_optional_value(command, "--sweep-band-strategy", config.get("sweep_band_strategy"))
    _append_optional_value(command, "--sweep-band-target-layer", config.get("sweep_band_target_layer"))
    _append_optional_value(
        command,
        "--detectability-consistency-signal",
        config.get("detectability_consistency_signal"),
    )
    _append_optional_value(
        command,
        "--detectability-confidence-signal",
        config.get("detectability_confidence_signal"),
    )
    command.extend((
        "--sweep-band-expand-radius",
        str(config.get("sweep_band_expand_radius") or 0),
        "--sweep-band-run-template",
        str(config.get("sweep_band_run_template") or "{cell}"),
        "--detectability-consistency-direction",
        str(config.get("detectability_consistency_direction") or "higher"),
        "--detectability-confidence-direction",
        str(config.get("detectability_confidence_direction") or "higher"),
    ))
    sweep_band_scales = _csv(config.get("sweep_band_scales", ()))
    if sweep_band_scales:
        command.extend(("--sweep-band-scales", sweep_band_scales))
    if config.get("attention_pathway") is True:
        command.append("--attention-pathway")
    if config.get("length_bucketed_batches") is False:
        command.append("--no-length-bucketed-batches")
    if config.get("offline") is True:
        command.append("--offline")
    if config.get("auto_batch_size") is False:
        command.append("--no-auto-batch-size")
    if config.get("cache_only") is True:
        command.append("--cache-only")
    if config.get("refresh_caches") is True:
        command.append("--refresh-caches")
    if config.get("refresh_scores") is True:
        command.append("--refresh-scores")
    return tuple(command)


def _multiple_testing_signals_for_command(config: Mapping[str, Any]) -> str:
    return _csv(config.get("multiple_testing_signals", ())) or _csv(config.get("signals", ()))


def _score_dump_for_cell(
    workflow_report: Mapping[str, Any],
    cell_name: str,
) -> str | None:
    for cell in _mapping_sequence(workflow_report.get("cells", ())):
        if _optional_str(cell.get("name")) != cell_name:
            continue
        score_dump = _mapping(cell.get("score_dump"))
        path = _optional_str(score_dump.get("path"))
        if path is not None:
            return path
    return None


def _model_scale_for_cell(
    config: Mapping[str, Any],
    cell_name: str,
) -> tuple[Mapping[str, Any] | None, Mapping[str, Any] | None]:
    for model in _mapping_sequence(config.get("models", ())):
        model_name = str(model.get("name") or "")
        for scale in _mapping_sequence(config.get("scales", ())):
            scale_name = str(scale.get("name") or "")
            if f"{model_name}-{scale_name}" == cell_name:
                return model, scale
    return None, None


def _format_scale(scale: Mapping[str, Any]) -> str:
    layers = ",".join(str(layer) for layer in _string_tuple(scale.get("sweep_layers", ())))
    return (
        f"{scale.get('name')}={int(scale.get('limit'))}:"
        f"{int(scale.get('manifold_questions'))}:"
        f"{int(scale.get('layer'))}:{layers}"
    )


def _matching_workflow_report(
    cell: Mapping[str, Any],
    workflow_reports: Sequence[tuple[Path, Mapping[str, Any]]],
) -> tuple[Path | None, Mapping[str, Any] | None]:
    run_name = _optional_str(cell.get("run"))
    cell_name = _optional_str(cell.get("cell")) or ""
    if run_name:
        for path, report in workflow_reports:
            if run_name in {_workflow_report_name(path, report), path.stem}:
                return path, report
    for path, report in workflow_reports:
        gate = _mapping(report.get("multiple_testing_gate"))
        cells = _mapping_sequence(gate.get("cells", ()))
        if any(_mapping(item).get("cell") == cell_name for item in cells):
            return path, report
    if len(workflow_reports) == 1:
        return workflow_reports[0]
    return None, None


def _load_frontier_workflow_reports(
    payload: Mapping[str, Any],
    *,
    source_path: Path,
    source_dir: Path,
) -> tuple[tuple[Path, Mapping[str, Any]], ...]:
    reports: list[tuple[Path, Mapping[str, Any]]] = []
    if payload.get("workflow") == "truthfulqa_frontier_workflow":
        reports.append((source_path, payload))
    inputs = _mapping(payload.get("inputs"))
    for item in _mapping_sequence(inputs.get("frontier_workflow_reports", ())):
        raw_path = item.get("path")
        if not raw_path:
            continue
        path = _resolve_report_path(Path(str(raw_path)), source_dir=source_dir)
        if path is None:
            continue
        if path.exists():
            report = _load_json_object(path)
            if report.get("workflow") == "truthfulqa_frontier_workflow":
                reports.append((path, report))
    return tuple(reports)


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
            "frontier_multiple_testing_rerun_queue": output_path,
        },
        root=manifest_path.parent,
        metadata={
            "runner": "plan_frontier_multiple_testing_reruns",
            "status": payload.get("status"),
            "source": str(source_path),
            "blocked_cell_count": summary.get("blocked_cell_count"),
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
    metadata = {
        "workflow": "frontier_multiple_testing_rerun_queue",
        "status": payload.get("status"),
        "source": payload.get("source"),
        "blocked_cell_count": summary.get("blocked_cell_count"),
        "command_count": summary.get("command_count"),
        "missing_command_count": summary.get("missing_command_count"),
        "artifact_manifest": None if manifest_path is None else str(manifest_path),
        "manifest_summary": None if manifest is None else _mapping(manifest.get("summary")),
    }
    ArtifactRegistry.load_json(registry_path).record_report(
        name=name,
        version=version,
        path=report_path,
        metadata=metadata,
    ).save_json()


def _workflow_report_name(path: Path, report: Mapping[str, Any]) -> str:
    for key in ("name", "run_name"):
        value = report.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    metadata = _mapping(report.get("metadata"))
    for key in ("name", "run_name"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return path.stem


def _unique_entries(entries: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    seen: set[tuple[str | None, str, str | None]] = set()
    unique = []
    for entry in entries:
        key = (
            _optional_str(entry.get("run")),
            str(entry.get("cell") or ""),
            _optional_str(entry.get("status")),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(dict(entry))
    return tuple(unique)


def _unique_cells(cells: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    seen: set[tuple[str | None, str, str | None]] = set()
    unique = []
    for cell in cells:
        key = (
            _optional_str(cell.get("run")),
            str(cell.get("cell") or ""),
            _optional_str(cell.get("status")),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(dict(cell))
    return tuple(unique)


def _load_json_object(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"JSON source must contain an object: {path}")
    return data


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _mapping_sequence(value: Any) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    return tuple(item for item in value if isinstance(item, Mapping))


def _string_tuple(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray)):
        return ()
    return tuple(str(item) for item in value if str(item))


def _csv(value: Any) -> str:
    return ",".join(_string_tuple(value))


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _status_from_pass(value: Any) -> str:
    if value is False:
        return "failed"
    if value is True:
        return "passed"
    return "unknown"


def _append_optional_value(command: list[str], flag: str, value: Any) -> None:
    text = _optional_str(value)
    if text is not None:
        command.extend((flag, text))


if __name__ == "__main__":
    main()
