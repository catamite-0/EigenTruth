"""Aggregate real-model belief-revision reports into a strict go/pause decision."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REQUIRED_METHODS = (
    "baseline_prompt",
    "self_correction_prompt",
    "rag_evidence_only",
    "eigentruth_revision_loop",
)


@dataclass(frozen=True)
class KillGatePolicy:
    minimum_models: int = 2
    minimum_examples_per_model: int = 20
    minimum_stubbornness_reduction: float = 0.10
    minimum_correction_gain: float = 0.10
    maximum_regressed_models: int = 0


def evaluate_kill_gate(
    reports: Sequence[Mapping[str, Any]],
    *,
    policy: KillGatePolicy = KillGatePolicy(),
) -> dict[str, Any]:
    model_rows = tuple(_model_row(report) for report in reports)
    failures: list[str] = []
    if len(model_rows) < policy.minimum_models:
        failures.append(f"need at least {policy.minimum_models} distinct model reports")
    model_ids = {row["model_id"] for row in model_rows}
    if len(model_ids) != len(model_rows):
        failures.append("model reports must use distinct model_id values")

    eligible = []
    for row in model_rows:
        if row["example_count"] < policy.minimum_examples_per_model:
            failures.append(
                f"{row['model_id']}: need at least {policy.minimum_examples_per_model} examples"
            )
        if row["missing_methods"]:
            failures.append(
                f"{row['model_id']}: missing methods {', '.join(row['missing_methods'])}"
            )
        if not row["missing_methods"] and row["example_count"] >= policy.minimum_examples_per_model:
            eligible.append(row)

    if failures:
        decision = "INSUFFICIENT_EVIDENCE"
    else:
        regressed = sum(1 for row in eligible if row["stubbornness_reduction"] < 0)
        passes = all(
            row["stubbornness_reduction"] >= policy.minimum_stubbornness_reduction
            and row["correction_gain"] >= policy.minimum_correction_gain
            for row in eligible
        )
        decision = (
            "CONTINUE_0_3"
            if passes and regressed <= policy.maximum_regressed_models
            else "PAUSE_PROJECT"
        )

    return {
        "schema_version": 1,
        "workflow": "belief_revision_kill_gate",
        "decision": decision,
        "policy": {
            "minimum_models": policy.minimum_models,
            "minimum_examples_per_model": policy.minimum_examples_per_model,
            "minimum_stubbornness_reduction": policy.minimum_stubbornness_reduction,
            "minimum_correction_gain": policy.minimum_correction_gain,
            "maximum_regressed_models": policy.maximum_regressed_models,
        },
        "models": model_rows,
        "failures": failures,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", type=Path, nargs="+")
    parser.add_argument("--json", type=Path)
    parser.add_argument("--minimum-models", type=int, default=2)
    parser.add_argument("--minimum-examples", type=int, default=20)
    parser.add_argument("--minimum-stubbornness-reduction", type=float, default=0.10)
    parser.add_argument("--minimum-correction-gain", type=float, default=0.10)
    args = parser.parse_args(argv)

    reports = [json.loads(path.read_text(encoding="utf-8")) for path in args.reports]
    result = evaluate_kill_gate(
        reports,
        policy=KillGatePolicy(
            minimum_models=args.minimum_models,
            minimum_examples_per_model=args.minimum_examples,
            minimum_stubbornness_reduction=args.minimum_stubbornness_reduction,
            minimum_correction_gain=args.minimum_correction_gain,
        ),
    )
    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if result["decision"] == "CONTINUE_0_3" else 2


def _model_row(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = _mapping(report.get("summary"))
    by_method = _mapping(summary.get("by_method"))
    missing = tuple(method for method in REQUIRED_METHODS if method not in by_method)
    self_correction = _mapping(by_method.get("self_correction_prompt"))
    eigentruth = _mapping(by_method.get("eigentruth_revision_loop"))
    example_count = int(summary.get("example_count", 0))
    return {
        "model_id": str(report.get("model_id", "")).strip() or "unknown-model",
        "example_count": example_count,
        "missing_methods": missing,
        "self_correction_stubbornness_rate": float(
            self_correction.get("stubbornness_rate", 0.0)
        ),
        "eigentruth_stubbornness_rate": float(eigentruth.get("stubbornness_rate", 0.0)),
        "stubbornness_reduction": float(self_correction.get("stubbornness_rate", 0.0))
        - float(eigentruth.get("stubbornness_rate", 0.0)),
        "self_correction_success_rate": float(
            self_correction.get("correction_success_rate", 0.0)
        ),
        "eigentruth_correction_success_rate": float(
            eigentruth.get("correction_success_rate", 0.0)
        ),
        "correction_gain": float(eigentruth.get("correction_success_rate", 0.0))
        - float(self_correction.get("correction_success_rate", 0.0)),
    }


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


if __name__ == "__main__":
    raise SystemExit(main())
