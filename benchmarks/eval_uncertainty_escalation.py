"""Summarize uncertainty-escalated verification-loop results.

The script is model-free. It consumes JSON/JSONL records produced from
``VerificationLoopResult.to_dict()`` or wrapper rows of the form
``{"label": 0, "result": {...}}`` and reports escalation, retrieval, decision
change, and label-conditioned acceptance metrics.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eigentruth.eval import uncertainty_escalation_report  # noqa: E402


def load_loop_result_records(path: Path) -> tuple[Mapping[str, Any], ...]:
    """Load verification-loop result records from JSON or JSONL."""
    if path.suffix.lower() == ".jsonl":
        records = []
        with open(path, encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                payload = json.loads(stripped)
                if not isinstance(payload, Mapping):
                    raise ValueError(f"{path}:{line_number} must contain a JSON object.")
                records.append(payload)
        return tuple(records)

    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, Mapping):
        raw_records = payload.get("records", payload.get("loop_results"))
        if raw_records is None:
            raw_records = (payload,)
    else:
        raw_records = payload
    if not isinstance(raw_records, list | tuple):
        raise ValueError("input JSON must be an object, an object with records/loop_results, or an array.")
    records = []
    for index, item in enumerate(raw_records):
        if not isinstance(item, Mapping):
            raise ValueError(f"input record {index} must be a JSON object.")
        records.append(item)
    return tuple(records)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, type=Path, help="JSON or JSONL loop-result records")
    parser.add_argument("--json", default=None, type=Path, help="optional output report path")
    parser.add_argument(
        "--label-key",
        default="label",
        help="wrapper field used for labels when rows do not already use 'label'",
    )
    args = parser.parse_args(argv)

    records = load_loop_result_records(args.results)
    normalized = tuple(_normalize_label_key(record, args.label_key) for record in records)
    report = uncertainty_escalation_report(normalized)

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, sort_keys=True)
            f.write("\n")
        print(f"Wrote uncertainty escalation report to {args.json}")
    else:
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _normalize_label_key(record: Mapping[str, Any], label_key: str) -> Mapping[str, Any]:
    if label_key == "label" or "label" in record or label_key not in record:
        return record
    normalized = dict(record)
    normalized["label"] = record[label_key]
    return normalized


if __name__ == "__main__":
    raise SystemExit(main())
