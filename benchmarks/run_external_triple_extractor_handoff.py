"""Run an external triple extractor command through the offline prediction gate.

The external extractor stays outside EigenTruth. This workflow writes a
label-free request JSONL from saved triple-extraction records, invokes a local
command that must write prediction JSON/JSONL, then evaluates those predictions
with the existing ``LookupTripleExtractor`` path.

The command is executed without a shell. Use ``{input}`` and ``{output}``
placeholders in the command string to receive the request JSONL and prediction
output path.
"""

from __future__ import annotations

import argparse
import json
import math
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.eval_triple_extraction import (  # noqa: E402
    load_triple_extraction_records,
    run_triple_extraction_eval,
)
from eigentruth.registry import ArtifactRegistry, ArtifactVerificationContext  # noqa: E402

DEFAULT_MIN_F1 = 0.90
DEFAULT_MIN_PRECISION = 0.90
DEFAULT_MIN_RECALL = 0.90
DEFAULT_MAX_FALSE_POSITIVE_RATE = 0.0
_OUTPUT_LIMIT = 4000


def run_external_triple_extractor_handoff(
    *,
    records_path: str | Path,
    extractor_command: str | Sequence[str],
    output_dir: str | Path,
    request_jsonl_path: str | Path | None = None,
    predictions_path: str | Path | None = None,
    eval_report_path: str | Path | None = None,
    workflow_report_path: str | Path | None = None,
    artifact_manifest_path: str | Path | None = None,
    verification_report_path: str | Path | None = None,
    registry_path: str | Path | None = None,
    name: str | None = None,
    version: str | None = None,
    include_metadata: bool = False,
    min_f1: float = DEFAULT_MIN_F1,
    min_precision: float = DEFAULT_MIN_PRECISION,
    min_recall: float = DEFAULT_MIN_RECALL,
    max_false_positive_rate: float = DEFAULT_MAX_FALSE_POSITIVE_RATE,
    max_examples: int = 20,
    command_timeout_seconds: float | None = None,
    manifest_fingerprint_workers: int = 1,
    compact_json: bool = False,
    fail_on_blocked: bool = False,
) -> dict[str, Any]:
    """Invoke an external extractor command and evaluate its predictions."""
    records_path = Path(records_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    request_jsonl = Path(request_jsonl_path or output_dir / "external-triple-extractor-requests.jsonl")
    predictions = Path(predictions_path or output_dir / "external-triple-predictions.jsonl")
    eval_report = Path(eval_report_path or output_dir / "external-triple-extraction-eval.json")
    workflow_report = Path(workflow_report_path or output_dir / "external-triple-extractor-handoff.json")
    manifest_path = None if artifact_manifest_path is None else Path(artifact_manifest_path)
    verification_path = None if verification_report_path is None else Path(verification_report_path)

    _validate_thresholds(
        min_f1=min_f1,
        min_precision=min_precision,
        min_recall=min_recall,
        max_false_positive_rate=max_false_positive_rate,
        max_examples=max_examples,
        manifest_fingerprint_workers=manifest_fingerprint_workers,
    )
    min_f1 = float(min_f1)
    min_precision = float(min_precision)
    min_recall = float(min_recall)
    max_false_positive_rate = float(max_false_positive_rate)
    max_examples = int(max_examples)
    manifest_fingerprint_workers = int(manifest_fingerprint_workers)
    records = load_triple_extraction_records(records_path)
    request_summary = _write_request_jsonl(request_jsonl, records, include_metadata=include_metadata)
    command_args = _format_command(
        extractor_command,
        input_path=request_jsonl,
        output_path=predictions,
    )
    predictions.parent.mkdir(parents=True, exist_ok=True)
    if predictions.exists():
        predictions.unlink()
    command_result = _run_command(
        command_args,
        timeout_seconds=command_timeout_seconds,
    )
    if not predictions.exists():
        raise FileNotFoundError(f"external extractor did not write predictions: {predictions}")
    eval_payload = run_triple_extraction_eval(
        records_path,
        extractor_name="external_predictions",
        predictions_path=predictions,
        max_examples=max_examples,
    )
    _write_json(eval_report, eval_payload, compact=compact_json)
    gate = _gate(
        eval_payload,
        min_f1=min_f1,
        min_precision=min_precision,
        min_recall=min_recall,
        max_false_positive_rate=max_false_positive_rate,
    )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "workflow": "external_triple_extractor_handoff",
        "status": "promote" if gate["passed"] else "blocked",
        "config": {
            "records_path": str(records_path),
            "include_metadata": bool(include_metadata),
            "min_f1": float(min_f1),
            "min_precision": float(min_precision),
            "min_recall": float(min_recall),
            "max_false_positive_rate": float(max_false_positive_rate),
            "max_examples": int(max_examples),
            "command_timeout_seconds": command_timeout_seconds,
        },
        "paths": {
            "requests": str(request_jsonl),
            "predictions": str(predictions),
            "eval_report": str(eval_report),
            "workflow_report": str(workflow_report),
            "artifact_manifest": None if manifest_path is None else str(manifest_path),
            "manifest_verification": None if verification_path is None else str(verification_path),
        },
        "request_summary": request_summary,
        "command": {
            "args": command_args,
            "returncode": command_result.returncode,
            "stdout": _bounded_text(command_result.stdout),
            "stderr": _bounded_text(command_result.stderr),
        },
        "eval_report": eval_payload,
        "gate": gate,
    }
    _write_json(workflow_report, payload, compact=compact_json)
    context = ArtifactVerificationContext()
    manifest = None
    verification = None
    if manifest_path is not None:
        manifest = _write_artifact_manifest(
            context=context,
            output_path=manifest_path,
            records_path=records_path,
            request_jsonl_path=request_jsonl,
            predictions_path=predictions,
            eval_report_path=eval_report,
            workflow_report_path=workflow_report,
            payload=payload,
            max_workers=manifest_fingerprint_workers,
        )
        payload["artifact_manifest_summary"] = manifest.get("summary")
        _write_json(workflow_report, payload, compact=compact_json)
        manifest = _write_artifact_manifest(
            context=context,
            output_path=manifest_path,
            records_path=records_path,
            request_jsonl_path=request_jsonl,
            predictions_path=predictions,
            eval_report_path=eval_report,
            workflow_report_path=workflow_report,
            payload=payload,
            max_workers=manifest_fingerprint_workers,
        )
        if verification_path is not None:
            verification = context.load_and_verify_artifact_manifest(
                manifest_path,
                recursive=True,
                max_workers=manifest_fingerprint_workers,
            ).to_dict()
            _write_json(verification_path, verification, compact=compact_json)
    _record_registry(
        registry_path=None if registry_path is None else Path(registry_path),
        name=name,
        version=version,
        workflow_report_path=workflow_report,
        manifest_path=manifest_path,
        verification_path=verification_path,
        payload=payload,
        verification=verification,
    )
    if fail_on_blocked and payload["status"] != "promote":
        raise SystemExit(1)
    return payload


def _write_request_jsonl(
    path: Path,
    records: Sequence[Mapping[str, Any]],
    *,
    include_metadata: bool,
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, record in enumerate(records):
        text = record.get("text", record.get("claim", record.get("statement")))
        if text is None or not str(text).strip():
            raise ValueError(f"record {index} must contain text, claim, or statement.")
        claim_id = record.get("claim_id", record.get("id", f"r{index}"))
        row: dict[str, Any] = {
            "claim_id": str(claim_id),
            "text": str(text),
        }
        if include_metadata:
            metadata = record.get("metadata")
            if isinstance(metadata, Mapping):
                row["metadata"] = dict(metadata)
        rows.append(row)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return {
        "record_count": len(rows),
        "include_metadata": bool(include_metadata),
        "contains_expected_triples": False,
    }


def _format_command(
    command: str | Sequence[str],
    *,
    input_path: Path,
    output_path: Path,
) -> list[str]:
    parts = shlex.split(command) if isinstance(command, str) else [str(item) for item in command]
    if not parts:
        raise ValueError("extractor_command must be non-empty.")
    formatted = [
        part.replace("{input}", str(input_path)).replace("{output}", str(output_path))
        for part in parts
    ]
    if not any("{input}" in part for part in parts):
        raise ValueError("extractor_command must include {input} placeholder.")
    if not any("{output}" in part for part in parts):
        raise ValueError("extractor_command must include {output} placeholder.")
    return formatted


def _run_command(
    command: Sequence[str],
    *,
    timeout_seconds: float | None,
) -> subprocess.CompletedProcess[str]:
    timeout = None if timeout_seconds is None else float(timeout_seconds)
    if timeout is not None and (not math.isfinite(timeout) or timeout <= 0.0):
        raise ValueError("command_timeout_seconds must be positive and finite when set.")
    return subprocess.run(
        list(command),
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _gate(
    eval_payload: Mapping[str, Any],
    *,
    min_f1: float,
    min_precision: float,
    min_recall: float,
    max_false_positive_rate: float,
) -> dict[str, Any]:
    report = _mapping(eval_payload.get("report"))
    failures: list[str] = []
    _check_min(failures, "f1", report.get("f1"), min_f1)
    _check_min(failures, "precision", report.get("precision"), min_precision)
    _check_min(failures, "recall", report.get("recall"), min_recall)
    _check_max(failures, "false_positive_rate", report.get("false_positive_rate"), max_false_positive_rate)
    return {
        "passed": not failures,
        "blocking_reasons": failures,
        "policy": {
            "min_f1": float(min_f1),
            "min_precision": float(min_precision),
            "min_recall": float(min_recall),
            "max_false_positive_rate": float(max_false_positive_rate),
        },
        "metrics": {
            "record_count": report.get("record_count"),
            "prediction_key_count": eval_payload.get("prediction_key_count"),
            "precision": report.get("precision"),
            "recall": report.get("recall"),
            "f1": report.get("f1"),
            "false_positive_rate": report.get("false_positive_rate"),
        },
    }


def _write_artifact_manifest(
    *,
    context: ArtifactVerificationContext,
    output_path: Path,
    records_path: Path,
    request_jsonl_path: Path,
    predictions_path: Path,
    eval_report_path: Path,
    workflow_report_path: Path,
    payload: Mapping[str, Any],
    max_workers: int,
) -> dict[str, Any]:
    gate = _mapping(payload.get("gate"))
    metrics = _mapping(gate.get("metrics"))
    manifest = context.build_artifact_manifest(
        {
            "records": records_path,
            "external_extractor_requests": request_jsonl_path,
            "external_predictions": predictions_path,
            "external_prediction_eval": eval_report_path,
            "workflow_report": workflow_report_path,
        },
        root=output_path.parent,
        metadata={
            "workflow": payload.get("workflow"),
            "status": payload.get("status"),
            "gate_passed": gate.get("passed"),
            "record_count": metrics.get("record_count"),
            "prediction_key_count": metrics.get("prediction_key_count"),
            "precision": metrics.get("precision"),
            "recall": metrics.get("recall"),
            "f1": metrics.get("f1"),
            "false_positive_rate": metrics.get("false_positive_rate"),
        },
        max_workers=max_workers,
    )
    _write_json(output_path, manifest, compact=False)
    return manifest


def _record_registry(
    *,
    registry_path: Path | None,
    name: str | None,
    version: str | None,
    workflow_report_path: Path,
    manifest_path: Path | None,
    verification_path: Path | None,
    payload: Mapping[str, Any],
    verification: Mapping[str, Any] | None,
) -> None:
    if registry_path is None:
        return
    if not name or not version:
        raise ValueError("--registry requires --name and --version.")
    gate = _mapping(payload.get("gate"))
    metrics = _mapping(gate.get("metrics"))
    ArtifactRegistry.load_json(registry_path).record_report(
        name=name,
        path=workflow_report_path,
        version=version,
        metadata={
            "workflow": payload.get("workflow"),
            "status": payload.get("status"),
            "gate_passed": gate.get("passed"),
            "record_count": metrics.get("record_count"),
            "prediction_key_count": metrics.get("prediction_key_count"),
            "f1": metrics.get("f1"),
            "false_positive_rate": metrics.get("false_positive_rate"),
            "artifact_manifest": None if manifest_path is None else str(manifest_path),
            "manifest_verification_report": None if verification_path is None else str(verification_path),
            "manifest_verified": None if verification is None else bool(verification.get("passed")),
        },
    ).save_json()


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _check_min(reasons: list[str], name: str, value: Any, threshold: float) -> None:
    numeric = _float_or_none(value)
    if numeric is None or numeric < threshold:
        reasons.append(f"{name} below {threshold}: {value!r}")


def _check_max(reasons: list[str], name: str, value: Any, threshold: float) -> None:
    numeric = _float_or_none(value)
    if numeric is None or numeric > threshold:
        reasons.append(f"{name} above {threshold}: {value!r}")


def _validate_thresholds(
    *,
    min_f1: float,
    min_precision: float,
    min_recall: float,
    max_false_positive_rate: float,
    max_examples: int,
    manifest_fingerprint_workers: int,
) -> None:
    for name, value in (
        ("min_f1", min_f1),
        ("min_precision", min_precision),
        ("min_recall", min_recall),
        ("max_false_positive_rate", max_false_positive_rate),
    ):
        numeric = float(value)
        if not math.isfinite(numeric) or not (0.0 <= numeric <= 1.0):
            raise ValueError(f"{name} must be in [0, 1].")
    if int(max_examples) < 0:
        raise ValueError("max_examples must be non-negative.")
    if int(manifest_fingerprint_workers) < 1:
        raise ValueError("manifest_fingerprint_workers must be positive.")


def _bounded_text(value: str | None) -> str:
    text = "" if value is None else str(value)
    if len(text) <= _OUTPUT_LIMIT:
        return text
    return text[:_OUTPUT_LIMIT] + "...[truncated]"


def _write_json(path: Path, payload: Mapping[str, Any], *, compact: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if compact:
        text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    else:
        text = json.dumps(payload, indent=2, sort_keys=True)
    path.write_text(text + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run from parsed CLI arguments."""
    return run_external_triple_extractor_handoff(
        records_path=args.records,
        extractor_command=args.extractor_command,
        output_dir=args.output_dir,
        request_jsonl_path=args.requests,
        predictions_path=args.predictions,
        eval_report_path=args.eval_report,
        workflow_report_path=args.json,
        artifact_manifest_path=args.artifact_manifest,
        verification_report_path=args.verification_report,
        registry_path=args.registry,
        name=args.name,
        version=args.version,
        include_metadata=bool(args.include_metadata),
        min_f1=args.min_f1,
        min_precision=args.min_precision,
        min_recall=args.min_recall,
        max_false_positive_rate=args.max_false_positive_rate,
        max_examples=args.max_examples,
        command_timeout_seconds=args.command_timeout_seconds,
        manifest_fingerprint_workers=args.manifest_fingerprint_workers,
        compact_json=bool(args.compact_json),
        fail_on_blocked=bool(args.fail_on_blocked),
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run an external triple extractor through offline predictions")
    parser.add_argument("--records", required=True, help="triple extraction records JSON/JSONL")
    parser.add_argument(
        "--extractor-command",
        required=True,
        help="command with {input} and {output} placeholders; executed without a shell",
    )
    parser.add_argument("--output-dir", default="artifacts/external-triple-extractor-handoff")
    parser.add_argument("--requests", default=None, help="optional request JSONL path")
    parser.add_argument("--predictions", default=None, help="optional prediction output path")
    parser.add_argument("--eval-report", default=None)
    parser.add_argument("--json", default=None, help="workflow report output path")
    parser.add_argument("--artifact-manifest", default=None)
    parser.add_argument("--verification-report", default=None)
    parser.add_argument("--registry", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--version", default=None)
    parser.add_argument("--include-metadata", action="store_true")
    parser.add_argument("--min-f1", type=float, default=DEFAULT_MIN_F1)
    parser.add_argument("--min-precision", type=float, default=DEFAULT_MIN_PRECISION)
    parser.add_argument("--min-recall", type=float, default=DEFAULT_MIN_RECALL)
    parser.add_argument("--max-false-positive-rate", type=float, default=DEFAULT_MAX_FALSE_POSITIVE_RATE)
    parser.add_argument("--max-examples", type=int, default=20)
    parser.add_argument("--command-timeout-seconds", type=float, default=None)
    parser.add_argument("--manifest-fingerprint-workers", type=int, default=1)
    parser.add_argument("--compact-json", action="store_true")
    parser.add_argument("--fail-on-blocked", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    payload = run(build_arg_parser().parse_args(argv))
    print(
        "external_triple_extractor_handoff="
        f"{payload['status']} f1={payload['gate']['metrics'].get('f1')}"
    )


if __name__ == "__main__":
    main()
