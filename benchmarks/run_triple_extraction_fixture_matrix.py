"""Compare generated triple-extraction fixture workflows across corpora.

This workflow promotes extractor templates only when they survive more than one
structured-fact corpus and cover a minimum predicate diversity. It is intended
as release evidence for moving from country-core KG facts toward broader
domain-specific fact extraction without adding learned extractor dependencies.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.run_triple_extraction_fixture_workflow import (  # noqa: E402
    TripleExtractionFixtureWorkflowConfig,
    run_triple_extraction_fixture_workflow,
)
from eigentruth.registry import build_artifact_manifest  # noqa: E402


@dataclass(frozen=True)
class TripleExtractionCorpusConfig:
    """One structured-fact corpus entry in the fixture matrix."""

    name: str
    fact_corpus_paths: Sequence[str | Path]

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        if not name:
            raise ValueError("corpus name must be non-empty.")
        paths = tuple(Path(path) for path in self.fact_corpus_paths)
        if not paths:
            raise ValueError(f"corpus {name!r} must include at least one fact corpus path.")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "fact_corpus_paths", paths)

    @property
    def slug(self) -> str:
        """Return a stable path-safe corpus slug."""
        return _slugify(self.name)


@dataclass(frozen=True)
class TripleExtractionFixtureMatrixConfig:
    """Configuration for cross-corpus triple-extraction fixture evidence."""

    corpora: Sequence[TripleExtractionCorpusConfig | Mapping[str, Any]]
    output_dir: str | Path
    max_facts: int | None = None
    max_examples: int = 20
    min_augmented_f1: float = 1.0
    require_f1_lift: bool = True
    adversarial_negatives_per_fact: int = 0
    max_adversarial_false_positive_rate: float = 0.0
    predicate_confusions_per_fact: int = 0
    min_predicate_confusion_f1: float = 1.0
    non_assertive_negatives_per_fact: int = 0
    max_non_assertive_false_positive_rate: float = 0.0
    min_corpora: int = 2
    min_distinct_predicates: int = 4
    compact_json: bool = False

    def __post_init__(self) -> None:
        corpora = tuple(_coerce_corpus(item) for item in self.corpora)
        if not corpora:
            raise ValueError("corpora must not be empty.")
        slugs = tuple(corpus.slug for corpus in corpora)
        if len(set(slugs)) != len(slugs):
            raise ValueError("corpus names must produce unique slugs.")
        object.__setattr__(self, "corpora", corpora)
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        if self.max_facts is not None and int(self.max_facts) <= 0:
            raise ValueError("max_facts must be positive when provided.")
        object.__setattr__(self, "max_facts", None if self.max_facts is None else int(self.max_facts))
        if int(self.max_examples) < 0:
            raise ValueError("max_examples must be non-negative.")
        object.__setattr__(self, "max_examples", int(self.max_examples))
        if int(self.adversarial_negatives_per_fact) < 0:
            raise ValueError("adversarial_negatives_per_fact must be non-negative.")
        object.__setattr__(
            self,
            "adversarial_negatives_per_fact",
            int(self.adversarial_negatives_per_fact),
        )
        max_adversarial_false_positive_rate = float(self.max_adversarial_false_positive_rate)
        if not (0.0 <= max_adversarial_false_positive_rate <= 1.0):
            raise ValueError("max_adversarial_false_positive_rate must be in [0, 1].")
        object.__setattr__(
            self,
            "max_adversarial_false_positive_rate",
            max_adversarial_false_positive_rate,
        )
        if int(self.predicate_confusions_per_fact) < 0:
            raise ValueError("predicate_confusions_per_fact must be non-negative.")
        object.__setattr__(
            self,
            "predicate_confusions_per_fact",
            int(self.predicate_confusions_per_fact),
        )
        min_predicate_confusion_f1 = float(self.min_predicate_confusion_f1)
        if not (0.0 <= min_predicate_confusion_f1 <= 1.0):
            raise ValueError("min_predicate_confusion_f1 must be in [0, 1].")
        object.__setattr__(self, "min_predicate_confusion_f1", min_predicate_confusion_f1)
        if int(self.non_assertive_negatives_per_fact) < 0:
            raise ValueError("non_assertive_negatives_per_fact must be non-negative.")
        object.__setattr__(
            self,
            "non_assertive_negatives_per_fact",
            int(self.non_assertive_negatives_per_fact),
        )
        max_non_assertive_false_positive_rate = float(self.max_non_assertive_false_positive_rate)
        if not (0.0 <= max_non_assertive_false_positive_rate <= 1.0):
            raise ValueError("max_non_assertive_false_positive_rate must be in [0, 1].")
        object.__setattr__(
            self,
            "max_non_assertive_false_positive_rate",
            max_non_assertive_false_positive_rate,
        )
        min_augmented_f1 = float(self.min_augmented_f1)
        if not (0.0 <= min_augmented_f1 <= 1.0):
            raise ValueError("min_augmented_f1 must be in [0, 1].")
        object.__setattr__(self, "min_augmented_f1", min_augmented_f1)
        if int(self.min_corpora) <= 0:
            raise ValueError("min_corpora must be positive.")
        object.__setattr__(self, "min_corpora", int(self.min_corpora))
        if int(self.min_distinct_predicates) < 0:
            raise ValueError("min_distinct_predicates must be non-negative.")
        object.__setattr__(self, "min_distinct_predicates", int(self.min_distinct_predicates))
        object.__setattr__(self, "require_f1_lift", bool(self.require_f1_lift))
        object.__setattr__(self, "compact_json", bool(self.compact_json))

    @property
    def summary_path(self) -> Path:
        return Path(self.output_dir) / "triple-extraction-fixture-matrix.json"

    @property
    def artifact_manifest_path(self) -> Path:
        return Path(self.output_dir) / "artifact-manifest.json"


def run_triple_extraction_fixture_matrix(config: TripleExtractionFixtureMatrixConfig) -> dict[str, Any]:
    """Run generated fixture workflows for each corpus and aggregate promotion evidence."""
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    corpus_results = []
    for corpus in config.corpora:
        corpus_dir = output_dir / corpus.slug
        summary = run_triple_extraction_fixture_workflow(
            TripleExtractionFixtureWorkflowConfig(
                fact_corpus_paths=corpus.fact_corpus_paths,
                output_dir=corpus_dir,
                max_facts=config.max_facts,
                max_examples=config.max_examples,
                min_augmented_f1=config.min_augmented_f1,
                require_f1_lift=config.require_f1_lift,
                adversarial_negatives_per_fact=config.adversarial_negatives_per_fact,
                max_adversarial_false_positive_rate=config.max_adversarial_false_positive_rate,
                predicate_confusions_per_fact=config.predicate_confusions_per_fact,
                min_predicate_confusion_f1=config.min_predicate_confusion_f1,
                non_assertive_negatives_per_fact=config.non_assertive_negatives_per_fact,
                max_non_assertive_false_positive_rate=config.max_non_assertive_false_positive_rate,
                compact_json=config.compact_json,
            )
        )
        corpus_results.append(_corpus_result(corpus=corpus, corpus_dir=corpus_dir, summary=summary))

    matrix = _matrix_summary(config=config, corpus_results=corpus_results)
    _write_json(config.summary_path, matrix, compact=config.compact_json)

    manifest = build_artifact_manifest(
        _manifest_artifacts(config=config, corpus_results=corpus_results),
        root=config.artifact_manifest_path.parent,
        metadata={
            "workflow": "triple_extraction_fixture_matrix",
            "status": matrix["status"],
            "n_corpora": matrix["n_corpora"],
            "promoted_corpora": matrix["promoted_corpora"],
            "distinct_predicate_count": matrix["distinct_predicate_count"],
            "min_distinct_predicates": config.min_distinct_predicates,
            "mean_best_f1": matrix["mean_best_f1"],
            "mean_baseline_f1": matrix["mean_baseline_f1"],
            "mean_f1_lift": matrix["mean_f1_lift"],
            "mean_best_adversarial_false_positive_rate": matrix[
                "mean_best_adversarial_false_positive_rate"
            ],
            "max_best_adversarial_false_positive_rate": matrix[
                "max_best_adversarial_false_positive_rate"
            ],
            "mean_best_predicate_confusion_f1": matrix["mean_best_predicate_confusion_f1"],
            "min_best_predicate_confusion_f1": matrix["min_best_predicate_confusion_f1"],
            "mean_best_non_assertive_false_positive_rate": matrix[
                "mean_best_non_assertive_false_positive_rate"
            ],
            "max_best_non_assertive_false_positive_rate": matrix[
                "max_best_non_assertive_false_positive_rate"
            ],
            "promotes_cross_corpus_extractor": matrix["status"] == "promote",
        },
    )
    _write_json(config.artifact_manifest_path, manifest, compact=False)
    print(
        "triple_extraction_fixture_matrix_ok "
        f"status={matrix['status']} "
        f"corpora={matrix['n_corpora']} "
        f"promoted={matrix['promoted_corpora']} "
        f"predicates={matrix['distinct_predicate_count']} "
        f"output={config.summary_path}"
    )
    return matrix


def _corpus_result(
    *,
    corpus: TripleExtractionCorpusConfig,
    corpus_dir: Path,
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    fixture_summary = _mapping(summary.get("fixture_summary"))
    by_predicate = _mapping(fixture_summary.get("by_predicate"))
    best_adversarial_report = _mapping(summary.get("best_adversarial_report"))
    best_predicate_confusion_report = _mapping(summary.get("best_predicate_confusion_report"))
    best_non_assertive_report = _mapping(summary.get("best_non_assertive_report"))
    return {
        "name": corpus.name,
        "slug": corpus.slug,
        "status": summary.get("status"),
        "fact_corpus_paths": tuple(str(path) for path in corpus.fact_corpus_paths),
        "workflow_summary_path": str(corpus_dir / "triple-extraction-workflow-summary.json"),
        "artifact_manifest_path": str(corpus_dir / "artifact-manifest.json"),
        "records_path": summary.get("records_path"),
        "patterns_path": summary.get("patterns_path"),
        "n_records": fixture_summary.get("n_records", 0),
        "n_facts": fixture_summary.get("n_facts", 0),
        "predicates": tuple(sorted(str(key) for key in by_predicate)),
        "baseline_f1": float(_mapping(summary.get("baseline_report")).get("f1", 0.0)),
        "best_extractor": summary.get("best_extractor"),
        "best_f1": float(_mapping(summary.get("best_report")).get("f1", 0.0)),
        "f1_lift": float(summary.get("f1_lift", 0.0)),
        "n_adversarial_negative_records": int(fixture_summary.get("n_adversarial_negative_records", 0)),
        "best_adversarial_false_positive_rate": float(
            best_adversarial_report.get("false_positive_rate", 0.0)
        ),
        "best_adversarial_false_positive_record_count": int(
            best_adversarial_report.get("false_positive_record_count", 0)
        ),
        "n_predicate_confusion_records": int(fixture_summary.get("n_predicate_confusion_records", 0)),
        "best_predicate_confusion_f1": float(best_predicate_confusion_report.get("f1", 0.0)),
        "best_predicate_confusion_exact_match_count": int(
            best_predicate_confusion_report.get("exact_match_count", 0)
        ),
        "n_non_assertive_negative_records": int(
            fixture_summary.get("n_non_assertive_negative_records", 0)
        ),
        "best_non_assertive_false_positive_rate": float(
            best_non_assertive_report.get("false_positive_rate", 0.0)
        ),
        "best_non_assertive_false_positive_record_count": int(
            best_non_assertive_report.get("false_positive_record_count", 0)
        ),
    }


def _matrix_summary(
    *,
    config: TripleExtractionFixtureMatrixConfig,
    corpus_results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    promoted = tuple(item for item in corpus_results if item.get("status") == "promote")
    distinct_predicates = tuple(sorted({
        predicate
        for item in corpus_results
        for predicate in item.get("predicates", ())
    }))
    failures = []
    if len(corpus_results) < config.min_corpora:
        failures.append({
            "gate": "min_corpora",
            "observed": len(corpus_results),
            "threshold": config.min_corpora,
        })
    if len(promoted) < config.min_corpora:
        failures.append({
            "gate": "promoted_corpora",
            "observed": len(promoted),
            "threshold": config.min_corpora,
        })
    blocked_corpora = tuple(item["name"] for item in corpus_results if item.get("status") != "promote")
    if blocked_corpora:
        failures.append({
            "gate": "all_corpora_promote",
            "blocked_corpora": blocked_corpora,
        })
    if len(distinct_predicates) < config.min_distinct_predicates:
        failures.append({
            "gate": "min_distinct_predicates",
            "observed": len(distinct_predicates),
            "threshold": config.min_distinct_predicates,
            "predicates": distinct_predicates,
        })
    status = "promote" if not failures else "blocked"
    return {
        "schema_version": 1,
        "workflow": "triple_extraction_fixture_matrix",
        "status": status,
        "promotion_gate": {
            "min_corpora": config.min_corpora,
            "min_distinct_predicates": config.min_distinct_predicates,
            "min_augmented_f1": config.min_augmented_f1,
            "require_f1_lift": config.require_f1_lift,
            "adversarial_negatives_per_fact": config.adversarial_negatives_per_fact,
            "max_adversarial_false_positive_rate": config.max_adversarial_false_positive_rate,
            "predicate_confusions_per_fact": config.predicate_confusions_per_fact,
            "min_predicate_confusion_f1": config.min_predicate_confusion_f1,
            "non_assertive_negatives_per_fact": config.non_assertive_negatives_per_fact,
            "max_non_assertive_false_positive_rate": config.max_non_assertive_false_positive_rate,
            "failures": tuple(failures),
        },
        "n_corpora": len(corpus_results),
        "promoted_corpora": len(promoted),
        "distinct_predicates": distinct_predicates,
        "distinct_predicate_count": len(distinct_predicates),
        "mean_baseline_f1": _mean(float(item["baseline_f1"]) for item in corpus_results),
        "mean_best_f1": _mean(float(item["best_f1"]) for item in corpus_results),
        "mean_f1_lift": _mean(float(item["f1_lift"]) for item in corpus_results),
        "mean_best_adversarial_false_positive_rate": _mean(
            float(item["best_adversarial_false_positive_rate"])
            for item in corpus_results
        ),
        "max_best_adversarial_false_positive_rate": max(
            (float(item["best_adversarial_false_positive_rate"]) for item in corpus_results),
            default=0.0,
        ),
        "mean_best_predicate_confusion_f1": _mean(
            float(item["best_predicate_confusion_f1"])
            for item in corpus_results
        ),
        "min_best_predicate_confusion_f1": min(
            (float(item["best_predicate_confusion_f1"]) for item in corpus_results),
            default=0.0,
        ),
        "mean_best_non_assertive_false_positive_rate": _mean(
            float(item["best_non_assertive_false_positive_rate"])
            for item in corpus_results
        ),
        "max_best_non_assertive_false_positive_rate": max(
            (float(item["best_non_assertive_false_positive_rate"]) for item in corpus_results),
            default=0.0,
        ),
        "corpora": tuple(dict(item) for item in corpus_results),
    }


def _manifest_artifacts(
    *,
    config: TripleExtractionFixtureMatrixConfig,
    corpus_results: Sequence[Mapping[str, Any]],
) -> dict[str, Path]:
    artifacts: dict[str, Path] = {"matrix_summary": config.summary_path}
    for result in corpus_results:
        slug = str(result["slug"])
        artifacts[f"corpus.{slug}.workflow_summary"] = Path(str(result["workflow_summary_path"]))
        artifacts[f"corpus.{slug}.artifact_manifest"] = Path(str(result["artifact_manifest_path"]))
        for idx, source_path in enumerate(result.get("fact_corpus_paths", ()), start=1):
            path = Path(str(source_path))
            artifacts[f"corpus.{slug}.source.{idx}.{path.stem}"] = path
    return artifacts


def _coerce_corpus(value: TripleExtractionCorpusConfig | Mapping[str, Any]) -> TripleExtractionCorpusConfig:
    if isinstance(value, TripleExtractionCorpusConfig):
        return value
    if not isinstance(value, Mapping):
        raise ValueError("corpora entries must be TripleExtractionCorpusConfig or mappings.")
    paths = value.get("fact_corpus_paths", value.get("paths", value.get("fact_corpus")))
    if isinstance(paths, (str, Path)):
        paths = (paths,)
    if not isinstance(paths, Sequence) or isinstance(paths, (bytes, bytearray)):
        raise ValueError("corpus mapping must include fact_corpus_paths.")
    return TripleExtractionCorpusConfig(
        name=str(value.get("name", "")),
        fact_corpus_paths=paths,
    )


def _parse_corpus_specs(specs: Sequence[str]) -> tuple[TripleExtractionCorpusConfig, ...]:
    grouped: dict[str, list[str]] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError("--corpus must use NAME=PATH format.")
        name, path = spec.split("=", 1)
        name = name.strip()
        path = path.strip()
        if not name or not path:
            raise ValueError("--corpus must use non-empty NAME=PATH values.")
        grouped.setdefault(name, []).append(path)
    return tuple(
        TripleExtractionCorpusConfig(name=name, fact_corpus_paths=tuple(paths))
        for name, paths in grouped.items()
    )


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9._-]+", "-", value.strip().casefold())
    slug = slug.strip("-._")
    if not slug:
        raise ValueError("corpus name must contain at least one path-safe character.")
    return slug


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _mean(values: Sequence[float] | Any) -> float:
    items = tuple(values)
    if not items:
        return 0.0
    return sum(items) / len(items)


def _write_json(path: Path, payload: Mapping[str, Any], *, compact: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    indent = None if compact else 2
    path.write_text(json.dumps(payload, indent=indent, sort_keys=True) + "\n", encoding="utf-8")


def _config_from_args(args: argparse.Namespace) -> TripleExtractionFixtureMatrixConfig:
    return TripleExtractionFixtureMatrixConfig(
        corpora=_parse_corpus_specs(tuple(args.corpus)),
        output_dir=args.output_dir,
        max_facts=args.max_facts,
        max_examples=args.max_examples,
        min_augmented_f1=args.min_augmented_f1,
        require_f1_lift=not bool(args.allow_no_lift),
        adversarial_negatives_per_fact=args.adversarial_negatives_per_fact,
        max_adversarial_false_positive_rate=args.max_adversarial_false_positive_rate,
        predicate_confusions_per_fact=args.predicate_confusions_per_fact,
        min_predicate_confusion_f1=args.min_predicate_confusion_f1,
        non_assertive_negatives_per_fact=args.non_assertive_negatives_per_fact,
        max_non_assertive_false_positive_rate=args.max_non_assertive_false_positive_rate,
        min_corpora=args.min_corpora,
        min_distinct_predicates=args.min_distinct_predicates,
        compact_json=bool(args.compact_json),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run cross-corpus triple extraction fixture matrix")
    parser.add_argument(
        "--corpus",
        action="append",
        required=True,
        help="corpus in NAME=PATH format; repeat the same NAME to group multiple paths",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-facts", type=int, default=None)
    parser.add_argument("--max-examples", type=int, default=20)
    parser.add_argument("--min-augmented-f1", type=float, default=1.0)
    parser.add_argument("--allow-no-lift", action="store_true")
    parser.add_argument("--adversarial-negatives-per-fact", type=int, default=0)
    parser.add_argument("--max-adversarial-false-positive-rate", type=float, default=0.0)
    parser.add_argument("--predicate-confusions-per-fact", type=int, default=0)
    parser.add_argument("--min-predicate-confusion-f1", type=float, default=1.0)
    parser.add_argument("--non-assertive-negatives-per-fact", type=int, default=0)
    parser.add_argument("--max-non-assertive-false-positive-rate", type=float, default=0.0)
    parser.add_argument("--min-corpora", type=int, default=2)
    parser.add_argument("--min-distinct-predicates", type=int, default=4)
    parser.add_argument("--compact-json", action="store_true")
    run_triple_extraction_fixture_matrix(_config_from_args(parser.parse_args()))


if __name__ == "__main__":
    main()
