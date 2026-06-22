"""Backfill TruthfulQA statement metadata into existing score dumps.

This is a no-model utility for older ``eval_truthfulqa.py --dump-scores``
artifacts created before score dumps included statement metadata. It rebuilds
the deterministic TruthfulQA eval split, validates label alignment, and can emit
an oracle claim/evidence fixture for verifier-ensemble upper-bound tests.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.eval_truthfulqa import (
    Statement,
    _batched_statements,
    _statement_to_dump,
    load_truthfulqa,
)


def ordered_eval_statements(
    eval_statements: Sequence[Statement],
    *,
    batch_size: int,
    length_bucketed: bool,
) -> tuple[Statement, ...]:
    """Return statements in the exact scoring order used by eval_truthfulqa."""
    return tuple(
        statement
        for batch in _batched_statements(eval_statements, batch_size, length_bucketed=length_bucketed)
        for statement in batch
    )


def backfill_statement_dump(
    dump: Mapping[str, Any],
    statements: Sequence[Statement],
) -> dict[str, Any]:
    """Return a score dump copy with validated statement metadata."""
    labels = tuple(int(label) for label in dump.get("labels", ()))
    if len(labels) != len(statements):
        raise ValueError(f"score dump has {len(labels)} labels but statement split has {len(statements)} rows.")
    statement_labels = tuple(int(statement.is_false) for statement in statements)
    if labels != statement_labels:
        mismatches = [
            idx for idx, (actual, expected) in enumerate(zip(labels, statement_labels))
            if actual != expected
        ]
        preview = ", ".join(str(idx) for idx in mismatches[:10])
        raise ValueError(f"score dump labels do not align with rebuilt TruthfulQA split at row(s): {preview}.")
    return {
        **dict(dump),
        "statements": [_statement_to_dump(statement) for statement in statements],
    }


def oracle_claim_fixture(
    statements: Sequence[Mapping[str, Any]],
    labels: Sequence[int],
) -> dict[str, Any]:
    """Build label-derived oracle evidence for controlled verifier upper bounds."""
    if len(statements) != len(labels):
        raise ValueError("statements and labels must have the same length.")
    records = []
    for idx, (statement, label) in enumerate(zip(statements, labels), start=1):
        text = str(statement.get("text") or statement.get("answer") or "").strip()
        if not text:
            raise ValueError(f"statement {idx} is missing text.")
        is_false = int(label) == 1
        evidence = f"Oracle label marks this claim false: {text}" if is_false else text
        record = {
            "claim": text,
            "claim_id": f"c{idx}",
            "retrieval_documents": [] if is_false else [evidence],
            "metadata": {
                "oracle_label": "false" if is_false else "true",
                "oracle_fixture": True,
            },
        }
        if is_false:
            record["refutations"] = {text: [evidence]}
        records.append(record)
    return {
        "schema_version": 1,
        "fixture_type": "truthfulqa_oracle_label_evidence",
        "description": (
            "Label-derived oracle evidence for verifier-ensemble upper-bound tests; "
            "not real retrieval evidence."
        ),
        "records": records,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    with open(args.scores, encoding="utf-8") as f:
        dump = json.load(f)

    config = dict(dump.get("config", {}))
    batch_size = args.batch_size if args.batch_size is not None else int(config.get("batch_size", 1))
    length_bucketed = (
        args.length_bucketed_batches
        if args.length_bucketed_batches is not None
        else bool(config.get("length_bucketed_batches", False))
    )
    _, _, eval_statements = load_truthfulqa(args.manifold_questions, args.limit)
    statements = ordered_eval_statements(
        eval_statements,
        batch_size=batch_size,
        length_bucketed=length_bucketed,
    )
    payload = backfill_statement_dump(dump, statements)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote statement-bearing score dump to {output_path}")

    if args.save_oracle_claims:
        fixture = oracle_claim_fixture(payload["statements"], payload["labels"])
        fixture_path = Path(args.save_oracle_claims)
        fixture_path.parent.mkdir(parents=True, exist_ok=True)
        with open(fixture_path, "w", encoding="utf-8") as f:
            json.dump(fixture, f, indent=2)
        print(f"Wrote oracle claim fixture to {fixture_path}")
        return {"scores": payload, "oracle_claims": fixture}
    return {"scores": payload}


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill TruthfulQA statement metadata into score dumps")
    parser.add_argument("--scores", required=True, help="existing eval_truthfulqa.py score dump")
    parser.add_argument("--output", required=True, help="path for statement-bearing score dump")
    parser.add_argument("--manifold-questions", type=int, required=True)
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=None,
                        help="override scoring batch size used for ordering")
    parser.add_argument("--length-bucketed-batches", action=argparse.BooleanOptionalAction, default=None,
                        help="override length-bucketed scoring order")
    parser.add_argument("--save-oracle-claims", default=None,
                        help="optional path for label-derived oracle claim/evidence fixture")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
