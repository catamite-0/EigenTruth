"""Evaluate evidence-grounded belief revision on text fixtures."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from benchmarks.lib.paths import ensure_repo_root_on_path
from eigentruth.json_utils import strict_json_dumps
from eigentruth.registry import build_artifact_manifest
from eigentruth.revision import (
    BeliefRevisionExample,
    BeliefRevisionResult,
    EvidenceGroundedRevisionEngine,
    evaluate_belief_revision_example,
)

REPO_ROOT = ensure_repo_root_on_path()
DEFAULT_EXAMPLES = (
    REPO_ROOT
    / "artifacts"
    / "baselines"
    / "belief_revision_text"
    / "belief-revision-seed.jsonl"
)
METHODS = (
    "baseline_prompt",
    "self_correction_prompt",
    "rag_evidence_only",
    "eigentruth_revision_loop",
)


def load_belief_revision_examples(path: str | Path) -> tuple[BeliefRevisionExample, ...]:
    examples: list[BeliefRevisionExample] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        payload = json.loads(stripped)
        if not isinstance(payload, Mapping):
            raise ValueError(f"belief revision fixture row {line_number} must be a JSON object.")
        examples.append(BeliefRevisionExample.from_dict(payload))
    return tuple(examples)


def build_belief_revision_report(
    *,
    examples: Sequence[BeliefRevisionExample],
    model_id: str,
    methods: Sequence[str] = METHODS,
) -> dict[str, Any]:
    engine = EvidenceGroundedRevisionEngine()
    results: list[BeliefRevisionResult] = []
    for example in examples:
        candidate_answers = _mapping(example.metadata.get("candidate_answers"))
        for method in methods:
            if method == "eigentruth_revision_loop":
                result = evaluate_belief_revision_example(
                    example,
                    model_id=model_id,
                    method=method,
                    engine=engine,
                )
            else:
                answer = str(candidate_answers.get(method, example.initial_answer))
                result = evaluate_belief_revision_example(
                    example,
                    model_id=model_id,
                    method=method,
                    answer=answer,
                )
            results.append(result)
    result_payloads = [result.to_dict() for result in results]
    summary = _summarize_results(result_payloads)
    return {
        "schema_version": 1,
        "workflow": "belief_revision_eval",
        "model_id": model_id,
        "methods": tuple(methods),
        "summary": summary,
        "results": result_payloads,
    }


def write_belief_revision_report(
    *,
    examples_path: str | Path = DEFAULT_EXAMPLES,
    model_id: str = "fixture-open-model",
    json_path: str | Path | None = None,
    artifact_manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    examples_path = Path(examples_path)
    examples = load_belief_revision_examples(examples_path)
    report = build_belief_revision_report(examples=examples, model_id=model_id)
    report_path: Path | None = None
    if json_path is not None:
        report_path = Path(json_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(strict_json_dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if artifact_manifest_path is not None:
        if report_path is None:
            raise ValueError("--artifact-manifest requires --json.")
        manifest = build_artifact_manifest(
            {"belief_revision_report": report_path, "examples": examples_path},
            root=REPO_ROOT,
            metadata={
                "workflow": "belief_revision_eval",
                "model_id": model_id,
                "example_count": len(examples),
            },
        )
        manifest_path = Path(artifact_manifest_path)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(strict_json_dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        report["artifact_manifest"] = str(manifest_path)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate text evidence-grounded belief revision")
    parser.add_argument("--examples", type=Path, default=DEFAULT_EXAMPLES)
    parser.add_argument("--model-id", default="fixture-open-model")
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--artifact-manifest", type=Path, default=None)
    args = parser.parse_args(argv)
    report = write_belief_revision_report(
        examples_path=args.examples,
        model_id=args.model_id,
        json_path=args.json,
        artifact_manifest_path=args.artifact_manifest,
    )
    summary = report["summary"]
    print(
        "belief_revision_eval_ok "
        f"examples={summary['example_count']} "
        f"eigentruth_stubbornness={summary['by_method']['eigentruth_revision_loop']['stubbornness_rate']:.3f}"
    )
    return 0


def _summarize_results(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_method: dict[str, dict[str, Any]] = {}
    for method in METHODS:
        rows = [row for row in results if row.get("method") == method]
        by_method[method] = _method_summary(rows)
    example_ids = {str(row.get("example_id")) for row in results if row.get("example_id") is not None}
    return {
        "example_count": len(example_ids) if example_ids else len(results) // max(1, len(METHODS)),
        "result_count": len(results),
        "by_method": by_method,
    }


def _method_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    if count == 0:
        return {
            "count": 0,
            "stubbornness_rate": 0.0,
            "unsupported_persistence_rate": 0.0,
            "evidence_uptake_rate": 0.0,
            "correction_success_rate": 0.0,
        }
    return {
        "count": count,
        "stubbornness_rate": _bool_rate(rows, "stubbornness"),
        "unsupported_persistence_rate": _bool_rate(rows, "unsupported_persistence"),
        "evidence_uptake_rate": _bool_rate(rows, "evidence_uptake"),
        "correction_success_rate": _bool_rate(rows, "correction_success"),
    }


def _bool_rate(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    return sum(1 for row in rows if bool(row.get(key))) / len(rows)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


if __name__ == "__main__":
    raise SystemExit(main())

