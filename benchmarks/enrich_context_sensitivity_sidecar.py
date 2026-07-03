"""Enrich verifier sidecars with context-sensitivity reports from paired logprobs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from eigentruth.eval.context_sensitivity import score_context_sensitivity


def load_verified_record_sidecar(path: str | Path) -> tuple[dict[str, Any], ...]:
    """Load verifier verified-record JSONL sidecar records."""
    records = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, Mapping):
                raise ValueError(f"verified-record line {line_number} must be a JSON object.")
            records.append(dict(payload))
    if not records:
        raise ValueError("verified-record sidecar must contain at least one record.")
    return tuple(records)


def load_paired_logprob_records(path: str | Path) -> tuple[dict[str, Any], ...]:
    """Load paired no-context/evidence-context token logprob records from JSON or JSONL."""
    source = Path(path)
    if source.suffix == ".jsonl":
        return _load_jsonl_records(source, record_name="paired-logprob")
    with source.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, Mapping):
        records = _first_sequence(
            payload.get("records"),
            payload.get("paired_logprobs"),
            payload.get("context_sensitivity_records"),
        )
        if records is None:
            return (dict(payload),)
        return _records_from_sequence(records, record_name="paired-logprob")
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        return _records_from_sequence(payload, record_name="paired-logprob")
    raise ValueError("paired-logprob JSON must be an object, an array of objects, or a records object.")


def enrich_context_sensitivity_sidecar(
    verified_records: Sequence[Mapping[str, Any]],
    paired_logprob_records: Sequence[Mapping[str, Any]],
    *,
    run_name: str | None = None,
    ratio_threshold: float = 1.25,
    shift_threshold: float = 0.25,
    min_abs_delta: float = 0.0,
    allow_missing: bool = False,
    overwrite: bool = False,
) -> tuple[tuple[dict[str, Any], ...], dict[str, Any]]:
    """Return verified records enriched with context-sensitivity reports."""
    ratio_threshold = _positive_float(ratio_threshold, name="ratio_threshold")
    shift_threshold = _non_negative_float(shift_threshold, name="shift_threshold")
    min_abs_delta = _non_negative_float(min_abs_delta, name="min_abs_delta")
    selected_run = None if run_name is None else str(run_name)
    default_run = _default_run(verified_records, run_name=selected_run)
    paired_by_key = _index_paired_logprob_records(paired_logprob_records, default_run=default_run)
    enriched: list[dict[str, Any]] = []
    enriched_summaries: list[Mapping[str, Any]] = []
    missing_keys: list[dict[str, Any]] = []
    matched_keys: set[tuple[str, int]] = set()

    for record in verified_records:
        output_record = dict(record)
        if selected_run is not None and str(record.get("run", "")) != selected_run:
            enriched.append(output_record)
            continue
        key = _record_key(record, default_run=default_run, record_name="verified-record")
        paired = paired_by_key.get(key)
        if paired is None:
            missing_keys.append({"run": key[0], "record_index": key[1]})
            if not allow_missing:
                raise ValueError(f"missing paired-logprob record for run={key[0]!r} record_index={key[1]}.")
            enriched.append(output_record)
            continue
        if not overwrite and (
            output_record.get("context_sensitivity") is not None
            or output_record.get("context_sensitivity_report") is not None
        ):
            raise ValueError(
                f"verified record run={key[0]!r} record_index={key[1]} already has context_sensitivity."
            )
        report = _context_sensitivity_report_from_paired_record(
            paired,
            key=key,
            ratio_threshold=ratio_threshold,
            shift_threshold=shift_threshold,
            min_abs_delta=min_abs_delta,
        )
        output_record["context_sensitivity"] = report
        enriched_summaries.append(report["summary"])
        matched_keys.add(key)
        enriched.append(output_record)

    unused_keys = sorted(key for key in paired_by_key if key not in matched_keys)
    report = {
        "schema_version": 1,
        "workflow": "context_sensitivity_sidecar_enrichment",
        "run_name": selected_run,
        "thresholds": {
            "ratio_threshold": ratio_threshold,
            "shift_threshold": shift_threshold,
            "min_abs_delta": min_abs_delta,
        },
        "input_record_count": len(verified_records),
        "paired_logprob_record_count": len(paired_logprob_records),
        "enriched_record_count": len(enriched_summaries),
        "missing_record_count": len(missing_keys),
        "unused_paired_logprob_count": len(unused_keys),
        "missing_records": missing_keys,
        "unused_paired_logprobs": [
            {"run": run, "record_index": record_index}
            for run, record_index in unused_keys
        ],
        "summary": _aggregate_context_sensitivity_summaries(enriched_summaries),
    }
    return tuple(enriched), report


def write_verified_records_jsonl(records: Sequence[Mapping[str, Any]], output_path: str | Path) -> None:
    """Write verified-record sidecar JSONL."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")


def build_report(
    *,
    verified_records_jsonl: str | Path,
    paired_logprobs: str | Path,
    output: str | Path,
    run_name: str | None = None,
    ratio_threshold: float = 1.25,
    shift_threshold: float = 0.25,
    min_abs_delta: float = 0.0,
    allow_missing: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Enrich a sidecar and return a compact enrichment report."""
    verified_records = load_verified_record_sidecar(verified_records_jsonl)
    paired_records = load_paired_logprob_records(paired_logprobs)
    enriched, report = enrich_context_sensitivity_sidecar(
        verified_records,
        paired_records,
        run_name=run_name,
        ratio_threshold=ratio_threshold,
        shift_threshold=shift_threshold,
        min_abs_delta=min_abs_delta,
        allow_missing=allow_missing,
        overwrite=overwrite,
    )
    write_verified_records_jsonl(enriched, output)
    return {
        **report,
        "verified_records_jsonl": str(verified_records_jsonl),
        "paired_logprobs": str(paired_logprobs),
        "output": str(output),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    """CLI entrypoint helper."""
    report = build_report(
        verified_records_jsonl=args.verified_records_jsonl,
        paired_logprobs=args.paired_logprobs,
        output=args.output,
        run_name=args.run_name,
        ratio_threshold=args.ratio_threshold,
        shift_threshold=args.shift_threshold,
        min_abs_delta=args.min_abs_delta,
        allow_missing=bool(args.allow_missing),
        overwrite=bool(args.overwrite),
    )
    if args.json:
        json_path = Path(args.json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote context-sensitivity enriched sidecar to {args.output}")
    return report


def _context_sensitivity_report_from_paired_record(
    paired: Mapping[str, Any],
    *,
    key: tuple[str, int],
    ratio_threshold: float,
    shift_threshold: float,
    min_abs_delta: float,
) -> dict[str, Any]:
    tokens = _tokens_from_paired_record(paired)
    metadata = {
        "source": "paired_logprobs",
        "run": key[0],
        "record_index": key[1],
    }
    claim_id = paired.get("claim_id")
    if claim_id is not None:
        metadata["claim_id"] = str(claim_id)
    paired_metadata = paired.get("metadata")
    if isinstance(paired_metadata, Mapping):
        metadata["paired_metadata"] = dict(paired_metadata)
    report = score_context_sensitivity(
        tokens,
        ratio_threshold=ratio_threshold,
        shift_threshold=shift_threshold,
        min_abs_delta=min_abs_delta,
        metadata=metadata,
    )
    return report.to_dict()


def _tokens_from_paired_record(record: Mapping[str, Any]) -> Sequence[Any]:
    context_payload = _mapping(record.get("context_sensitivity"))
    tokens = _first_sequence(
        record.get("tokens"),
        record.get("context_sensitivity_tokens"),
        record.get("paired_token_logprobs"),
        context_payload.get("tokens"),
        context_payload.get("context_sensitivity_tokens"),
        context_payload.get("paired_token_logprobs"),
    )
    if tokens is None:
        raise ValueError(
            "paired-logprob record must include tokens, context_sensitivity_tokens, or paired_token_logprobs."
        )
    return tokens


def _index_paired_logprob_records(
    records: Sequence[Mapping[str, Any]],
    *,
    default_run: str | None,
) -> dict[tuple[str, int], Mapping[str, Any]]:
    indexed: dict[tuple[str, int], Mapping[str, Any]] = {}
    for record in records:
        key = _record_key(record, default_run=default_run, record_name="paired-logprob")
        if key in indexed:
            raise ValueError(f"duplicate paired-logprob record for run={key[0]!r} record_index={key[1]}.")
        indexed[key] = record
    return indexed


def _default_run(records: Sequence[Mapping[str, Any]], *, run_name: str | None) -> str | None:
    if run_name is not None:
        return run_name
    runs = {str(record.get("run", "")) for record in records if str(record.get("run", ""))}
    if len(runs) == 1:
        return next(iter(runs))
    return None


def _record_key(record: Mapping[str, Any], *, default_run: str | None, record_name: str) -> tuple[str, int]:
    run = str(record.get("run", "")).strip()
    if not run:
        if default_run is None:
            raise ValueError(f"{record_name} record run is required when multiple or unknown runs are present.")
        run = default_run
    record_index = _record_index(record, record_name=record_name)
    return run, record_index


def _record_index(record: Mapping[str, Any], *, record_name: str) -> int:
    value = record.get("record_index")
    if isinstance(value, bool) or value is None:
        raise ValueError(f"{record_name} record_index must be an integer.")
    try:
        numeric = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{record_name} record_index must be an integer.") from exc
    if numeric < 0:
        raise ValueError(f"{record_name} record_index must be non-negative.")
    return numeric


def _aggregate_context_sensitivity_summaries(summaries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    flagged_rates = [
        _finite_float(summary.get("flagged_rate", 0.0), name="summary.flagged_rate")
        for summary in summaries
    ]
    max_shifts = [
        _finite_float(
            summary.get("max_unsupported_context_shift", 0.0),
            name="summary.max_unsupported_context_shift",
        )
        for summary in summaries
    ]
    mean_shifts = [
        _finite_float(
            summary.get("mean_unsupported_context_shift", 0.0),
            name="summary.mean_unsupported_context_shift",
        )
        for summary in summaries
    ]
    max_ratios = [
        _finite_float(
            summary.get("max_context_sensitivity_ratio", 0.0),
            name="summary.max_context_sensitivity_ratio",
        )
        for summary in summaries
    ]
    return {
        "record_count": len(summaries),
        "mean_flagged_rate": _mean(flagged_rates),
        "max_flagged_rate": max(flagged_rates) if flagged_rates else 0.0,
        "max_unsupported_context_shift": max(max_shifts) if max_shifts else 0.0,
        "mean_unsupported_context_shift": _mean(mean_shifts),
        "max_context_sensitivity_ratio": max(max_ratios) if max_ratios else 0.0,
    }


def _load_jsonl_records(path: Path, *, record_name: str) -> tuple[dict[str, Any], ...]:
    records = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, Mapping):
                raise ValueError(f"{record_name} line {line_number} must be a JSON object.")
            records.append(dict(payload))
    if not records:
        raise ValueError(f"{record_name} JSONL must contain at least one record.")
    return tuple(records)


def _records_from_sequence(values: Sequence[Any], *, record_name: str) -> tuple[dict[str, Any], ...]:
    records = []
    for index, value in enumerate(values):
        if not isinstance(value, Mapping):
            raise ValueError(f"{record_name} record {index} must be a JSON object.")
        records.append(dict(value))
    if not records:
        raise ValueError(f"{record_name} records must contain at least one record.")
    return tuple(records)


def _first_sequence(*values: Any) -> Sequence[Any] | None:
    for value in values:
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return value
    return None


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _positive_float(value: Any, *, name: str) -> float:
    numeric = _finite_float(value, name=name)
    if numeric <= 0.0:
        raise ValueError(f"{name} must be positive.")
    return numeric


def _non_negative_float(value: Any, *, name: str) -> float:
    numeric = _finite_float(value, name=name)
    if numeric < 0.0:
        raise ValueError(f"{name} must be non-negative.")
    return numeric


def _finite_float(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number, not bool.")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number.") from exc
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite.")
    return numeric


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enrich eval_verifier_ensemble JSONL sidecars with context-sensitivity reports"
    )
    parser.add_argument("--verified-records-jsonl", required=True, help="input eval_verifier_ensemble sidecar JSONL")
    parser.add_argument(
        "--paired-logprobs",
        required=True,
        help="JSON/JSONL records with run, record_index, and paired token logprobs",
    )
    parser.add_argument("--output", required=True, help="output enriched verified-record sidecar JSONL")
    parser.add_argument("--run-name", default=None, help="optional run to enrich when sidecar has multiple runs")
    parser.add_argument("--ratio-threshold", type=float, default=1.25)
    parser.add_argument("--shift-threshold", type=float, default=0.25)
    parser.add_argument("--min-abs-delta", type=float, default=0.0)
    parser.add_argument("--allow-missing", action="store_true", help="copy records without matching paired logprobs")
    parser.add_argument("--overwrite", action="store_true", help="replace existing context_sensitivity payloads")
    parser.add_argument("--json", default=None, help="optional summary report path")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
