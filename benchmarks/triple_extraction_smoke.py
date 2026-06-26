"""Run a deterministic smoke check for configurable triple extraction.

This uses the bundled labeled fixture to verify that regex-augmented extractor
paths improve exact triple extraction over the default dependency-free rules.
It does not promote any learned extractor; it only gates the local adapter slot.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

run_triple_extraction_eval = importlib.import_module(
    "benchmarks.eval_triple_extraction"
).run_triple_extraction_eval

DEFAULT_RECORDS_PATH = REPO_ROOT / "benchmarks" / "fixtures" / "triple_extraction_records.json"
DEFAULT_PATTERNS_PATH = REPO_ROOT / "benchmarks" / "fixtures" / "triple_extraction_regex_patterns.json"
EXTRACTORS = ("rule_based", "regex_rule_based", "composite")


def build_triple_extraction_smoke(
    output_dir: Path,
    *,
    records_path: Path = DEFAULT_RECORDS_PATH,
    patterns_path: Path = DEFAULT_PATTERNS_PATH,
    max_examples: int = 20,
) -> dict[str, Any]:
    """Evaluate bundled extractor variants and assert the augmented paths win."""
    output_dir.mkdir(parents=True, exist_ok=True)
    reports = {
        extractor: run_triple_extraction_eval(
            records_path,
            extractor_name=extractor,
            patterns_path=patterns_path if extractor != "rule_based" else None,
            max_examples=max_examples,
        )
        for extractor in EXTRACTORS
    }
    for extractor, payload in reports.items():
        _write_json(output_dir / f"{extractor}-triple-extraction-report.json", payload)

    rule_report = reports["rule_based"]["report"]
    regex_report = reports["regex_rule_based"]["report"]
    composite_report = reports["composite"]["report"]
    if regex_report["recall"] <= rule_report["recall"]:
        raise AssertionError("regex_rule_based extractor did not improve recall over rule_based.")
    if regex_report["f1"] <= rule_report["f1"]:
        raise AssertionError("regex_rule_based extractor did not improve F1 over rule_based.")
    if composite_report["f1"] != regex_report["f1"]:
        raise AssertionError("composite extractor diverged from regex_rule_based fixture result.")
    if regex_report["f1"] < 1.0:
        raise AssertionError("regex_rule_based extractor should exactly cover the bundled fixture.")

    result = {
        "workflow": "triple_extraction_smoke",
        "records_path": str(records_path),
        "patterns_path": str(patterns_path),
        "output_dir": str(output_dir),
        "reports": reports,
        "summary": {
            "rule_based_f1": rule_report["f1"],
            "regex_rule_based_f1": regex_report["f1"],
            "composite_f1": composite_report["f1"],
            "f1_lift": regex_report["f1"] - rule_report["f1"],
            "recall_lift": regex_report["recall"] - rule_report["recall"],
        },
    }
    _write_json(output_dir / "triple-extraction-smoke-summary.json", result)
    return result


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run the triple extraction smoke check."""
    records_path = Path(args.records)
    patterns_path = Path(args.patterns)
    if args.output_dir:
        result = build_triple_extraction_smoke(
            Path(args.output_dir),
            records_path=records_path,
            patterns_path=patterns_path,
            max_examples=args.max_examples,
        )
    else:
        with tempfile.TemporaryDirectory(prefix="eigentruth-triple-extraction-") as tmpdir:
            result = build_triple_extraction_smoke(
                Path(tmpdir),
                records_path=records_path,
                patterns_path=patterns_path,
                max_examples=args.max_examples,
            )
    summary = result["summary"]
    print(
        "triple_extraction_smoke_ok "
        f"rule_f1={summary['rule_based_f1']:.3f} "
        f"regex_f1={summary['regex_rule_based_f1']:.3f} "
        f"f1_lift={summary['f1_lift']:.3f}"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deterministic triple extraction smoke checks")
    parser.add_argument("--records", default=str(DEFAULT_RECORDS_PATH))
    parser.add_argument("--patterns", default=str(DEFAULT_PATTERNS_PATH))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-examples", type=int, default=20)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
