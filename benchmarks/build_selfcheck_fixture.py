"""Build self-consistency claim fixtures from sampled generations.

This bridges sampled continuation artifacts to ``eval_verifier_ensemble.py``.
It is intentionally dependency-free: sample generation happens upstream
(``eval_truthfulqa.py --dump-inside-samples`` or an external sampler), while this
script only aligns samples to score-dump statements and writes claim fixtures.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eigentruth.eval.score_dump import load_score_dump as _load_validated_score_dump

SAMPLE_KEYS = (
    "selfcheck_samples",
    "sampled_responses",
    "samples",
    "inside_sample_texts",
    "sample_texts",
    "responses",
    "continuations",
)


def load_score_dump(path: Path) -> dict[str, Any]:
    """Load and validate a statement-bearing score dump."""
    return _load_validated_score_dump(
        path,
        allow_missing_scores=True,
        require_statements=True,
    ).to_mapping()


def load_sample_payloads(paths: Sequence[Path]) -> tuple[Mapping[str, Any] | Sequence[Any], ...]:
    """Load external sampled-generation payloads."""
    return tuple(_load_sample_payload(path) for path in paths)


def build_selfcheck_fixture(
    dump: Mapping[str, Any],
    sample_payloads: Sequence[Mapping[str, Any] | Sequence[Any]] = (),
    *,
    min_samples: int = 2,
    include_empty_records: bool = True,
) -> dict[str, Any]:
    """Build a verifier-ensemble fixture containing self-check samples."""
    if min_samples < 1:
        raise ValueError("min_samples must be >= 1.")
    labels = tuple(int(label) for label in dump.get("labels", ()))
    statements = tuple(dict(statement) for statement in dump.get("statements", ()))
    if len(labels) != len(statements):
        raise ValueError("labels and statements must have the same length.")

    indexed_payloads = tuple(_index_sample_payload(payload, expected_count=len(labels)) for payload in sample_payloads)
    records = []
    total_samples = 0
    dropped_records = 0
    for idx, (label, statement) in enumerate(zip(labels, statements)):
        claim_text = _statement_text(statement)
        claim_id = str(statement.get("claim_id") or f"c{idx + 1}")
        samples = _dedupe_samples(_samples_for_record(dump, statement, indexed_payloads, idx=idx, claim_id=claim_id))
        has_enough = len(samples) >= min_samples
        if not include_empty_records and not has_enough:
            dropped_records += 1
            continue
        total_samples += len(samples)
        records.append({
            "claim": claim_text,
            "claim_id": claim_id,
            "claim_metadata": dict(statement.get("metadata", {})),
            "selfcheck_samples": list(samples),
            "metadata": {
                "index": idx,
                "score_label": label,
                "statement": statement,
                "selfcheck": {
                    "n_samples": len(samples),
                    "meets_min_samples": has_enough,
                    "min_samples": min_samples,
                    "source": "score_dump_or_external_samples",
                },
            },
        })

    records_with_samples = sum(1 for record in records if record["selfcheck_samples"])
    records_meeting_min = sum(
        1 for record in records
        if len(record["selfcheck_samples"]) >= min_samples
    )
    return {
        "schema_version": 1,
        "fixture_type": "selfcheck_samples",
        "description": (
            "Self-consistency fixture built from sampled generations. Labels are copied only "
            "for audit metadata; verification uses claim text and sampled responses only."
        ),
        "selfcheck": {
            "min_samples": min_samples,
            "sample_keys": SAMPLE_KEYS,
        },
        "summary": {
            "n_records": len(records),
            "records_with_samples": records_with_samples,
            "records_meeting_min_samples": records_meeting_min,
            "records_dropped_below_min_samples": dropped_records,
            "total_samples": total_samples,
            "average_samples_per_record": float(total_samples) / len(records) if records else 0.0,
        },
        "records": records,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run the CLI command."""
    dump = load_score_dump(Path(args.scores))
    sample_payloads = load_sample_payloads(tuple(Path(path) for path in args.samples or ()))
    fixture = build_selfcheck_fixture(
        dump,
        sample_payloads,
        min_samples=args.min_samples,
        include_empty_records=not args.drop_empty_records,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(fixture, f, indent=2)
    summary = fixture["summary"]
    print(
        f"Wrote self-check fixture to {output_path} "
        f"({summary['records_meeting_min_samples']}/{summary['n_records']} records meeting min samples)"
    )
    return fixture


def _load_sample_payload(path: Path) -> Mapping[str, Any] | Sequence[Any]:
    if path.suffix.lower() == ".jsonl":
        records = []
        with open(path, encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    item = {"response": line, "source": f"{path}:{line_no}"}
                records.append(item)
        return records
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, (Mapping, Sequence)) or isinstance(payload, (str, bytes, bytearray)):
        raise ValueError("sample payload must be a JSON object or list.")
    return payload


def _index_sample_payload(payload: Mapping[str, Any] | Sequence[Any], *, expected_count: int) -> dict[str, Any]:
    by_index: dict[int, list[Any]] = {}
    by_claim_id: dict[str, list[Any]] = {}
    if isinstance(payload, Mapping):
        _add_top_level_sample_arrays(payload, by_index, expected_count=expected_count)
        records = payload.get("records")
        if isinstance(records, Sequence) and not isinstance(records, (str, bytes, bytearray)):
            _add_record_samples(records, by_index, by_claim_id)
        for key, value in payload.items():
            if key in {"records", "metadata", "config", "summary", *SAMPLE_KEYS}:
                continue
            samples = _as_sample_sequence(value)
            if samples:
                by_claim_id.setdefault(str(key), []).extend(samples)
    else:
        if len(payload) == expected_count:
            for idx, value in enumerate(payload):
                samples = _samples_from_payload_item(value)
                if samples:
                    by_index.setdefault(idx, []).extend(samples)
        _add_record_samples(payload, by_index, by_claim_id)
    return {"by_index": by_index, "by_claim_id": by_claim_id}


def _add_top_level_sample_arrays(
    payload: Mapping[str, Any],
    by_index: dict[int, list[Any]],
    *,
    expected_count: int,
) -> None:
    for key in SAMPLE_KEYS:
        if key not in payload:
            continue
        value = payload[key]
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            if len(value) != expected_count:
                continue
            for idx, item in enumerate(value):
                samples = _as_sample_sequence(item)
                if samples:
                    by_index.setdefault(idx, []).extend(samples)


def _add_record_samples(
    records: Sequence[Any],
    by_index: dict[int, list[Any]],
    by_claim_id: dict[str, list[Any]],
) -> None:
    for item in records:
        if not isinstance(item, Mapping):
            continue
        samples = _samples_from_mapping(item)
        if not samples:
            continue
        claim_id = item.get("claim_id")
        if claim_id is not None:
            by_claim_id.setdefault(str(claim_id), []).extend(samples)
        index = item.get("index", _mapping_path(item, ("metadata", "index")))
        if index is not None:
            by_index.setdefault(int(index), []).extend(samples)


def _samples_for_record(
    dump: Mapping[str, Any],
    statement: Mapping[str, Any],
    indexed_payloads: Sequence[Mapping[str, Any]],
    *,
    idx: int,
    claim_id: str,
) -> tuple[Any, ...]:
    samples: list[Any] = []
    samples.extend(_samples_from_mapping(statement))
    for key in SAMPLE_KEYS:
        value = dump.get(key)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)) and len(value) > idx:
            samples.extend(_as_sample_sequence(value[idx]))
    for payload in indexed_payloads:
        samples.extend(payload["by_index"].get(idx, ()))
        samples.extend(payload["by_claim_id"].get(claim_id, ()))
    return tuple(samples)


def _samples_from_payload_item(value: Any) -> tuple[Any, ...]:
    if isinstance(value, Mapping):
        samples = _samples_from_mapping(value)
        return samples if samples else _as_sample_sequence(value)
    return _as_sample_sequence(value)


def _samples_from_mapping(value: Mapping[str, Any]) -> tuple[Any, ...]:
    for key in SAMPLE_KEYS:
        if key in value:
            return _as_sample_sequence(value[key])
    return ()


def _as_sample_sequence(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Mapping):
        return (dict(value),)
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        samples = []
        for item in value:
            if isinstance(item, str):
                samples.append(item)
            elif isinstance(item, Mapping):
                samples.append(dict(item))
            else:
                samples.append(str(item))
        return tuple(samples)
    return (str(value),)


def _dedupe_samples(samples: Sequence[Any]) -> tuple[Any, ...]:
    deduped = []
    seen = set()
    for sample in samples:
        key = _sample_text(sample)
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(sample)
    return tuple(deduped)


def _sample_text(sample: Any) -> str:
    if isinstance(sample, str):
        return sample.strip()
    if isinstance(sample, Mapping):
        return str(sample.get("text", sample.get("content", sample.get("response", "")))).strip()
    return str(sample).strip()


def _statement_text(statement: Mapping[str, Any]) -> str:
    text = str(statement.get("claim") or statement.get("text") or statement.get("answer") or "").strip()
    if not text:
        raise ValueError("statement record is missing claim/text/answer.")
    return text


def _mapping_path(value: Mapping[str, Any], path: Sequence[str]) -> Any:
    current: Any = value
    for part in path:
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def main() -> None:
    parser = argparse.ArgumentParser(description="Build self-check fixture for verifier ensemble benchmarks")
    parser.add_argument("--scores", required=True, help="statement-bearing score dump")
    parser.add_argument("--samples", action="append", default=None,
                        help="optional sampled responses JSON/JSONL; repeatable")
    parser.add_argument("--output", required=True, help="path to write claim fixture JSON")
    parser.add_argument("--min-samples", type=int, default=2,
                        help="minimum sample count recorded in summary as ready for self-consistency")
    parser.add_argument("--drop-empty-records", action="store_true",
                        help="drop records that do not meet --min-samples")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
