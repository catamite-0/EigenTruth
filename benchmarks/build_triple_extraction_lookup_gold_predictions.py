"""Build lookup-gold triple predictions from labeled extraction records.

This helper turns a generated ``triple-extraction-records.json`` fixture into a
deterministic external-prediction mapping. It is intended for release evidence
replays where the matrix workflow needs an auditable local prediction artifact
without adding a learned extractor dependency.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.eval_triple_extraction import load_triple_extraction_records  # noqa: E402


def build_lookup_gold_predictions(records_path: str | Path) -> dict[str, list[dict[str, Any]]]:
    """Return a claim-id keyed mapping from fixture records to expected triples."""
    records = load_triple_extraction_records(records_path)
    predictions: dict[str, list[dict[str, Any]]] = {}
    for index, record in enumerate(records):
        key = _prediction_key(record, index=index)
        if key in predictions:
            raise ValueError(f"duplicate prediction key {key!r} in {records_path}.")
        predictions[key] = [_triple_payload(item) for item in _expected_triples(record)]
    return predictions


def _prediction_key(record: Mapping[str, Any], *, index: int) -> str:
    raw = record.get("claim_id", record.get("id", f"r{index}"))
    text = str(raw).strip()
    if not text:
        raise ValueError(f"record {index} must contain a non-empty claim_id or id.")
    return f"claim_id:{text}"


def _expected_triples(record: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    raw = record.get("expected_triples", record.get("triples", record.get("expected", ())))
    if raw is None:
        return ()
    if isinstance(raw, Mapping):
        return (raw,)
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        triples = []
        for index, item in enumerate(raw):
            if not isinstance(item, Mapping):
                raise ValueError(f"expected triple {index} must be a mapping.")
            triples.append(item)
        return tuple(triples)
    raise ValueError("expected triples must be a mapping or sequence of mappings.")


def _triple_payload(item: Mapping[str, Any]) -> dict[str, Any]:
    subject = item.get("subject")
    predicate = item.get("predicate")
    object_value = item.get("object", item.get("object_text"))
    if subject is None or predicate is None or object_value is None:
        raise ValueError("expected triples must contain subject, predicate, and object/object_text.")
    payload = {
        "subject": str(subject),
        "predicate": str(predicate),
        "object": str(object_value),
    }
    if item.get("confidence") is not None:
        payload["confidence"] = float(item["confidence"])
    if isinstance(item.get("metadata"), Mapping) and item["metadata"]:
        payload["metadata"] = dict(item["metadata"])
    return payload


def _write_json(path: Path, payload: Mapping[str, Any], *, compact: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    indent = None if compact else 2
    path.write_text(json.dumps(payload, indent=indent, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build lookup-gold external triple predictions from labeled records"
    )
    parser.add_argument("--records", required=True, help="input triple-extraction records JSON/JSONL")
    parser.add_argument("--output", required=True, help="output prediction mapping JSON")
    parser.add_argument("--compact-json", action="store_true")
    args = parser.parse_args(argv)

    predictions = build_lookup_gold_predictions(args.records)
    _write_json(Path(args.output), predictions, compact=bool(args.compact_json))
    triple_count = sum(len(items) for items in predictions.values())
    print(
        "triple_extraction_lookup_gold_predictions_ok "
        f"records={len(predictions)} "
        f"triples={triple_count} "
        f"output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
