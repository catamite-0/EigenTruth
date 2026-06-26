"""Build and evaluate generated triple-extraction fixtures.

This workflow turns structured fact corpora into labeled extraction records,
runs extractor variants, and writes a release-evidence-ready summary plus
artifact manifest. It is dependency-free and reuses the lower-level builder and
evaluator scripts.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.build_triple_extraction_fixture import (  # noqa: E402
    build_default_regex_pattern_payload,
    build_input_provenance,
    build_triple_extraction_fixture,
    load_fact_records,
)
from benchmarks.eval_triple_extraction import run_triple_extraction_eval  # noqa: E402
from eigentruth.registry import build_artifact_manifest  # noqa: E402

EXTRACTORS = ("rule_based", "regex_rule_based", "composite")


@dataclass(frozen=True)
class TripleExtractionFixtureWorkflowConfig:
    """Configuration for generated triple-extraction fixture workflow."""

    fact_corpus_paths: Sequence[str | Path]
    output_dir: str | Path
    max_facts: int | None = None
    max_examples: int = 20
    min_augmented_f1: float = 1.0
    require_f1_lift: bool = True
    compact_json: bool = False

    def __post_init__(self) -> None:
        fact_paths = tuple(Path(path) for path in self.fact_corpus_paths)
        if not fact_paths:
            raise ValueError("fact_corpus_paths must not be empty.")
        object.__setattr__(self, "fact_corpus_paths", fact_paths)
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        if self.max_facts is not None and int(self.max_facts) <= 0:
            raise ValueError("max_facts must be positive when provided.")
        object.__setattr__(self, "max_facts", None if self.max_facts is None else int(self.max_facts))
        if int(self.max_examples) < 0:
            raise ValueError("max_examples must be non-negative.")
        object.__setattr__(self, "max_examples", int(self.max_examples))
        min_augmented_f1 = float(self.min_augmented_f1)
        if not (0.0 <= min_augmented_f1 <= 1.0):
            raise ValueError("min_augmented_f1 must be in [0, 1].")
        object.__setattr__(self, "min_augmented_f1", min_augmented_f1)
        object.__setattr__(self, "require_f1_lift", bool(self.require_f1_lift))
        object.__setattr__(self, "compact_json", bool(self.compact_json))

    @property
    def records_path(self) -> Path:
        return Path(self.output_dir) / "triple-extraction-records.json"

    @property
    def patterns_path(self) -> Path:
        return Path(self.output_dir) / "triple-extraction-regex-patterns.json"

    @property
    def summary_path(self) -> Path:
        return Path(self.output_dir) / "triple-extraction-workflow-summary.json"

    @property
    def artifact_manifest_path(self) -> Path:
        return Path(self.output_dir) / "artifact-manifest.json"


def run_triple_extraction_fixture_workflow(
    config: TripleExtractionFixtureWorkflowConfig,
) -> dict[str, Any]:
    """Build generated fixture records, evaluate extractors, and summarize readiness."""
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_fact_records(config.fact_corpus_paths)
    fixture = build_triple_extraction_fixture(records, max_facts=config.max_facts)
    fixture["input_provenance"] = build_input_provenance(
        config.fact_corpus_paths,
        max_facts=config.max_facts,
    )
    pattern_payload = build_default_regex_pattern_payload()
    _write_json(config.records_path, fixture, compact=config.compact_json)
    _write_json(config.patterns_path, pattern_payload, compact=config.compact_json)

    reports = {}
    report_paths = {}
    for extractor in EXTRACTORS:
        report_path = output_dir / f"{extractor}-triple-extraction-report.json"
        reports[extractor] = run_triple_extraction_eval(
            config.records_path,
            extractor_name=extractor,
            patterns_path=config.patterns_path if extractor != "rule_based" else None,
            max_examples=config.max_examples,
        )
        _write_json(report_path, reports[extractor], compact=config.compact_json)
        report_paths[extractor] = str(report_path)

    summary = _workflow_summary(
        config=config,
        fixture=fixture,
        pattern_payload=pattern_payload,
        reports=reports,
        report_paths=report_paths,
    )
    _write_json(config.summary_path, summary, compact=config.compact_json)

    manifest = build_artifact_manifest(
        {
            "records": config.records_path,
            "patterns": config.patterns_path,
            "workflow_summary": config.summary_path,
            **{f"report.{name}": Path(path) for name, path in report_paths.items()},
            **{f"fact_corpus.{idx}.{path.stem}": path for idx, path in enumerate(config.fact_corpus_paths, start=1)},
        },
        root=config.artifact_manifest_path.parent,
        metadata={
            "workflow": "triple_extraction_fixture_workflow",
            "status": summary["status"],
            "n_records": summary["fixture_summary"]["n_records"],
            "pattern_count": summary["pattern_count"],
            "best_extractor": summary["best_extractor"],
            "best_f1": summary["best_report"]["f1"],
            "baseline_f1": summary["baseline_report"]["f1"],
            "f1_lift": summary["f1_lift"],
            "promotes_augmented_extractor": summary["status"] == "promote",
        },
    )
    _write_json(config.artifact_manifest_path, manifest, compact=False)
    print(
        "triple_extraction_fixture_workflow_ok "
        f"status={summary['status']} "
        f"records={summary['fixture_summary']['n_records']} "
        f"best={summary['best_extractor']} "
        f"f1={summary['best_report']['f1']:.3f} "
        f"output={config.summary_path}"
    )
    return summary


def _workflow_summary(
    *,
    config: TripleExtractionFixtureWorkflowConfig,
    fixture: Mapping[str, Any],
    pattern_payload: Mapping[str, Any],
    reports: Mapping[str, Mapping[str, Any]],
    report_paths: Mapping[str, str],
) -> dict[str, Any]:
    baseline_report = dict(reports["rule_based"]["report"])
    augmented_candidates = {
        name: dict(payload["report"])
        for name, payload in reports.items()
        if name != "rule_based"
    }
    best_extractor = max(
        augmented_candidates,
        key=lambda name: (
            float(augmented_candidates[name]["f1"]),
            float(augmented_candidates[name]["recall"]),
            float(augmented_candidates[name]["precision"]),
        ),
    )
    best_report = augmented_candidates[best_extractor]
    f1_lift = float(best_report["f1"]) - float(baseline_report["f1"])
    failures = []
    if float(best_report["f1"]) < config.min_augmented_f1:
        failures.append({
            "gate": "min_augmented_f1",
            "observed": float(best_report["f1"]),
            "threshold": config.min_augmented_f1,
        })
    if config.require_f1_lift and f1_lift <= 0.0:
        failures.append({
            "gate": "require_f1_lift",
            "observed": f1_lift,
            "threshold": 0.0,
        })
    status = "promote" if not failures else "blocked"
    return {
        "workflow": "triple_extraction_fixture_workflow",
        "status": status,
        "promotion_gate": {
            "min_augmented_f1": config.min_augmented_f1,
            "require_f1_lift": config.require_f1_lift,
            "failures": tuple(failures),
        },
        "records_path": str(config.records_path),
        "patterns_path": str(config.patterns_path),
        "report_paths": dict(report_paths),
        "fact_corpus_paths": tuple(str(path) for path in config.fact_corpus_paths),
        "fixture_summary": dict(fixture["summary"]),
        "pattern_count": len(pattern_payload["patterns"]),
        "baseline_extractor": "rule_based",
        "baseline_report": baseline_report,
        "best_extractor": best_extractor,
        "best_report": best_report,
        "f1_lift": f1_lift,
        "reports": {name: dict(payload["report"]) for name, payload in reports.items()},
    }


def _write_json(path: Path, payload: Mapping[str, Any], *, compact: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    indent = None if compact else 2
    path.write_text(json.dumps(payload, indent=indent, sort_keys=True) + "\n", encoding="utf-8")


def _config_from_args(args: argparse.Namespace) -> TripleExtractionFixtureWorkflowConfig:
    return TripleExtractionFixtureWorkflowConfig(
        fact_corpus_paths=tuple(args.fact_corpus),
        output_dir=args.output_dir,
        max_facts=args.max_facts,
        max_examples=args.max_examples,
        min_augmented_f1=args.min_augmented_f1,
        require_f1_lift=not bool(args.allow_no_lift),
        compact_json=bool(args.compact_json),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run generated triple extraction fixture workflow")
    parser.add_argument("--fact-corpus", action="append", required=True, help="fact corpus JSON/JSONL; repeatable")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-facts", type=int, default=None)
    parser.add_argument("--max-examples", type=int, default=20)
    parser.add_argument("--min-augmented-f1", type=float, default=1.0)
    parser.add_argument("--allow-no-lift", action="store_true")
    parser.add_argument("--compact-json", action="store_true")
    run_triple_extraction_fixture_workflow(_config_from_args(parser.parse_args()))


if __name__ == "__main__":
    main()
