"""Run an external triple extractor command across a fixture matrix.

This workflow is the matrix-level counterpart to
``run_external_triple_extractor_handoff.py``. It builds deterministic labeled
fixtures for each corpus, sends label-free requests to one or more local
external extractor commands, then feeds the generated prediction files into the
existing cross-corpus triple-extraction matrix gate.
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

from benchmarks.run_external_triple_extractor_handoff import (  # noqa: E402
    DEFAULT_MAX_FALSE_POSITIVE_RATE,
    DEFAULT_MIN_F1,
    DEFAULT_MIN_PRECISION,
    DEFAULT_MIN_RECALL,
    run_external_triple_extractor_handoff,
)
from benchmarks.run_triple_extraction_fixture_matrix import (  # noqa: E402
    TripleExtractionCorpusConfig,
    TripleExtractionFixtureMatrixConfig,
    run_triple_extraction_fixture_matrix,
)
from benchmarks.run_triple_extraction_fixture_workflow import (  # noqa: E402
    TripleExtractionFixtureWorkflowConfig,
    run_triple_extraction_fixture_workflow,
)
from eigentruth.registry import ArtifactRegistry, ArtifactVerificationContext  # noqa: E402

DEFAULT_WORKFLOW_NAME = "external-triple-extractor-matrix-handoff"


@dataclass(frozen=True)
class ExternalTripleExtractorMatrixHandoffConfig:
    """Configuration for external extractor command handoff over a matrix."""

    corpora: Sequence[TripleExtractionCorpusConfig | Mapping[str, Any]]
    extractor_commands: Mapping[str, str | Sequence[str]]
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
    ambiguity_negatives_per_fact: int = 0
    max_ambiguity_false_positive_rate: float = 0.0
    temporal_negatives_per_fact: int = 0
    max_temporal_false_positive_rate: float = 0.0
    metalinguistic_negatives_per_fact: int = 0
    max_metalinguistic_false_positive_rate: float = 0.0
    min_corpora: int = 2
    min_distinct_predicates: int = 4
    include_metadata: bool = False
    min_external_f1: float = DEFAULT_MIN_F1
    min_external_precision: float = DEFAULT_MIN_PRECISION
    min_external_recall: float = DEFAULT_MIN_RECALL
    max_external_false_positive_rate: float = DEFAULT_MAX_FALSE_POSITIVE_RATE
    command_timeout_seconds: float | None = None
    manifest_fingerprint_workers: int = 1
    compact_json: bool = False
    registry_path: str | Path | None = None
    name: str = DEFAULT_WORKFLOW_NAME
    version: str = "0.1"
    workflow_report_path: str | Path | None = None
    artifact_manifest_path: str | Path | None = None
    verification_report_path: str | Path | None = None
    fail_on_blocked: bool = False

    def __post_init__(self) -> None:
        corpora = tuple(_coerce_corpus(corpus) for corpus in self.corpora)
        if not corpora:
            raise ValueError("corpora must not be empty.")
        slugs = tuple(corpus.slug for corpus in corpora)
        if len(set(slugs)) != len(slugs):
            raise ValueError("corpus names must produce unique slugs.")
        commands = {}
        for name, command in self.extractor_commands.items():
            safe_name = _safe_name(name)
            if safe_name in commands:
                raise ValueError(f"duplicate external extractor command name: {safe_name}")
            if isinstance(command, str):
                if not command.strip():
                    raise ValueError("external extractor command must be non-empty.")
                commands[safe_name] = command
            else:
                parts = tuple(str(part) for part in command)
                if not parts:
                    raise ValueError("external extractor command must be non-empty.")
                commands[safe_name] = parts
        if not commands:
            raise ValueError("extractor_commands must not be empty.")
        object.__setattr__(self, "corpora", corpora)
        object.__setattr__(self, "extractor_commands", commands)
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        object.__setattr__(self, "workflow_report_path", self._defaulted_report_path())
        object.__setattr__(self, "artifact_manifest_path", self._defaulted_manifest_path())
        object.__setattr__(self, "verification_report_path", self._defaulted_verification_path())
        object.__setattr__(
            self,
            "registry_path",
            None if self.registry_path is None else Path(self.registry_path),
        )
        _validate_non_negative_int("max_examples", self.max_examples)
        _validate_positive_int("min_corpora", self.min_corpora)
        _validate_non_negative_int("min_distinct_predicates", self.min_distinct_predicates)
        _validate_positive_int("manifest_fingerprint_workers", self.manifest_fingerprint_workers)
        if self.max_facts is not None:
            _validate_positive_int("max_facts", self.max_facts)
        for name, value in (
            ("min_augmented_f1", self.min_augmented_f1),
            ("max_adversarial_false_positive_rate", self.max_adversarial_false_positive_rate),
            ("min_predicate_confusion_f1", self.min_predicate_confusion_f1),
            ("max_non_assertive_false_positive_rate", self.max_non_assertive_false_positive_rate),
            ("max_ambiguity_false_positive_rate", self.max_ambiguity_false_positive_rate),
            ("max_temporal_false_positive_rate", self.max_temporal_false_positive_rate),
            ("max_metalinguistic_false_positive_rate", self.max_metalinguistic_false_positive_rate),
            ("min_external_f1", self.min_external_f1),
            ("min_external_precision", self.min_external_precision),
            ("min_external_recall", self.min_external_recall),
            ("max_external_false_positive_rate", self.max_external_false_positive_rate),
        ):
            _validate_unit_interval(name, value)
        for name, value in (
            ("adversarial_negatives_per_fact", self.adversarial_negatives_per_fact),
            ("predicate_confusions_per_fact", self.predicate_confusions_per_fact),
            ("non_assertive_negatives_per_fact", self.non_assertive_negatives_per_fact),
            ("ambiguity_negatives_per_fact", self.ambiguity_negatives_per_fact),
            ("temporal_negatives_per_fact", self.temporal_negatives_per_fact),
            ("metalinguistic_negatives_per_fact", self.metalinguistic_negatives_per_fact),
        ):
            _validate_non_negative_int(name, value)
        if not str(self.name).strip():
            raise ValueError("name must be non-empty.")
        if not str(self.version).strip():
            raise ValueError("version must be non-empty.")

    def _defaulted_report_path(self) -> Path:
        if self.workflow_report_path is not None:
            return Path(self.workflow_report_path)
        return Path(self.output_dir) / "external-triple-extractor-matrix-handoff.json"

    def _defaulted_manifest_path(self) -> Path:
        if self.artifact_manifest_path is not None:
            return Path(self.artifact_manifest_path)
        return Path(self.output_dir) / "artifact-manifest.json"

    def _defaulted_verification_path(self) -> Path | None:
        if self.verification_report_path is None:
            return None
        return Path(self.verification_report_path)


def run_external_triple_extractor_matrix_handoff(
    config: ExternalTripleExtractorMatrixHandoffConfig,
) -> dict[str, Any]:
    """Run the full external extractor matrix handoff workflow."""
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    corpus_entries = []
    prediction_paths: dict[str, dict[str, Path]] = {}
    for corpus in config.corpora:
        prep_summary = _build_fixture_for_corpus(config=config, corpus=corpus)
        handoff_summaries = {}
        corpus_predictions = {}
        for extractor_name, command in config.extractor_commands.items():
            handoff = run_external_triple_extractor_handoff(
                records_path=prep_summary["records_path"],
                extractor_command=command,
                output_dir=output_dir / "external-handoffs" / corpus.slug / extractor_name,
                artifact_manifest_path=(
                    output_dir
                    / "external-handoffs"
                    / corpus.slug
                    / extractor_name
                    / "artifact-manifest.json"
                ),
                verification_report_path=(
                    output_dir
                    / "external-handoffs"
                    / corpus.slug
                    / extractor_name
                    / "manifest-verification.json"
                ),
                name=f"{config.name}-{corpus.slug}-{extractor_name}",
                version=config.version,
                include_metadata=config.include_metadata,
                min_f1=config.min_external_f1,
                min_precision=config.min_external_precision,
                min_recall=config.min_external_recall,
                max_false_positive_rate=config.max_external_false_positive_rate,
                max_examples=config.max_examples,
                command_timeout_seconds=config.command_timeout_seconds,
                manifest_fingerprint_workers=config.manifest_fingerprint_workers,
                compact_json=config.compact_json,
                fail_on_blocked=False,
            )
            predictions_path = Path(str(_mapping(handoff.get("paths")).get("predictions")))
            corpus_predictions[extractor_name] = predictions_path
            handoff_summaries[extractor_name] = _handoff_summary(handoff)
        prediction_paths[corpus.slug] = corpus_predictions
        corpus_entries.append({
            "name": corpus.name,
            "slug": corpus.slug,
            "fact_corpus_paths": tuple(str(path) for path in corpus.fact_corpus_paths),
            "fixture_summary_path": prep_summary["summary_path"],
            "fixture_records_path": prep_summary["records_path"],
            "external_handoffs": handoff_summaries,
        })

    matrix_config = TripleExtractionFixtureMatrixConfig(
        corpora=config.corpora,
        output_dir=output_dir / "matrix",
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
        ambiguity_negatives_per_fact=config.ambiguity_negatives_per_fact,
        max_ambiguity_false_positive_rate=config.max_ambiguity_false_positive_rate,
        temporal_negatives_per_fact=config.temporal_negatives_per_fact,
        max_temporal_false_positive_rate=config.max_temporal_false_positive_rate,
        metalinguistic_negatives_per_fact=config.metalinguistic_negatives_per_fact,
        max_metalinguistic_false_positive_rate=config.max_metalinguistic_false_positive_rate,
        min_corpora=config.min_corpora,
        min_distinct_predicates=config.min_distinct_predicates,
        external_prediction_paths_by_corpus=prediction_paths,
        compact_json=config.compact_json,
    )
    matrix = run_triple_extraction_fixture_matrix(matrix_config)
    gate = _matrix_handoff_gate(config=config, matrix=matrix, corpus_entries=corpus_entries)
    payload = {
        "schema_version": 1,
        "workflow": "external_triple_extractor_matrix_handoff",
        "status": "promote" if gate["passed"] else "blocked",
        "config": _config_payload(config),
        "paths": {
            "workflow_report": str(config.workflow_report_path),
            "artifact_manifest": str(config.artifact_manifest_path),
            "manifest_verification": (
                None if config.verification_report_path is None else str(config.verification_report_path)
            ),
            "matrix_report": str(matrix_config.summary_path),
            "matrix_manifest": str(matrix_config.artifact_manifest_path),
        },
        "gate": gate,
        "matrix": _matrix_summary(matrix=matrix, matrix_config=matrix_config),
        "corpora": corpus_entries,
    }
    _write_json(Path(config.workflow_report_path), payload, compact=config.compact_json)
    verification = _write_manifest_and_optional_verification(config=config, payload=payload)
    _record_registry(config=config, payload=payload, verification=verification)
    if config.fail_on_blocked and payload["status"] != "promote":
        raise SystemExit(1)
    print(
        "external_triple_extractor_matrix_handoff="
        f"{payload['status']} "
        f"corpora={matrix.get('n_corpora')} "
        f"external_predictions={matrix.get('external_prediction_count')} "
        f"output={config.workflow_report_path}"
    )
    return payload


def _build_fixture_for_corpus(
    *,
    config: ExternalTripleExtractorMatrixHandoffConfig,
    corpus: TripleExtractionCorpusConfig,
) -> dict[str, str]:
    prep_dir = Path(config.output_dir) / "fixture-prep" / corpus.slug
    summary = run_triple_extraction_fixture_workflow(
        TripleExtractionFixtureWorkflowConfig(
            fact_corpus_paths=corpus.fact_corpus_paths,
            output_dir=prep_dir,
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
            ambiguity_negatives_per_fact=config.ambiguity_negatives_per_fact,
            max_ambiguity_false_positive_rate=config.max_ambiguity_false_positive_rate,
            temporal_negatives_per_fact=config.temporal_negatives_per_fact,
            max_temporal_false_positive_rate=config.max_temporal_false_positive_rate,
            metalinguistic_negatives_per_fact=config.metalinguistic_negatives_per_fact,
            max_metalinguistic_false_positive_rate=config.max_metalinguistic_false_positive_rate,
            compact_json=config.compact_json,
        )
    )
    return {
        "summary_path": str(prep_dir / "triple-extraction-workflow-summary.json"),
        "records_path": str(summary["records_path"]),
        "artifact_manifest_path": str(prep_dir / "artifact-manifest.json"),
    }


def _matrix_handoff_gate(
    *,
    config: ExternalTripleExtractorMatrixHandoffConfig,
    matrix: Mapping[str, Any],
    corpus_entries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    failures = []
    if matrix.get("status") != "promote":
        failures.append({
            "gate": "matrix_promotes",
            "status": matrix.get("status"),
            "matrix_failures": _mapping(matrix.get("promotion_gate")).get("failures", ()),
        })
    blocked_handoffs = []
    for corpus in corpus_entries:
        for extractor_name, handoff in _mapping(corpus.get("external_handoffs")).items():
            if _mapping(handoff).get("status") != "promote":
                blocked_handoffs.append({
                    "corpus": corpus.get("slug"),
                    "extractor": extractor_name,
                    "status": _mapping(handoff).get("status"),
                    "blocking_reasons": _mapping(_mapping(handoff).get("gate")).get(
                        "blocking_reasons",
                        (),
                    ),
                })
    if blocked_handoffs:
        failures.append({"gate": "external_handoffs_promote", "blocked_handoffs": blocked_handoffs})
    expected_prediction_count = len(config.corpora) * len(config.extractor_commands)
    external_prediction_count = int(matrix.get("external_prediction_count", 0))
    if external_prediction_count < expected_prediction_count:
        failures.append({
            "gate": "external_prediction_count",
            "observed": external_prediction_count,
            "threshold": expected_prediction_count,
        })
    external_prediction_corpora = tuple(matrix.get("external_prediction_corpora", ()))
    if len(external_prediction_corpora) < len(config.corpora):
        failures.append({
            "gate": "external_prediction_corpora",
            "observed": len(external_prediction_corpora),
            "threshold": len(config.corpora),
        })
    mean_best_external_f1 = _float_or_none(matrix.get("mean_best_external_f1"))
    if mean_best_external_f1 is None or mean_best_external_f1 < float(config.min_external_f1):
        failures.append({
            "gate": "mean_best_external_f1",
            "observed": matrix.get("mean_best_external_f1"),
            "threshold": float(config.min_external_f1),
        })
    return {
        "passed": not failures,
        "failures": failures,
        "policy": {
            "expected_external_prediction_count": expected_prediction_count,
            "expected_external_prediction_corpora": len(config.corpora),
            "min_external_f1": float(config.min_external_f1),
            "min_external_precision": float(config.min_external_precision),
            "min_external_recall": float(config.min_external_recall),
            "max_external_false_positive_rate": float(config.max_external_false_positive_rate),
        },
        "metrics": {
            "external_prediction_count": external_prediction_count,
            "external_prediction_corpora": external_prediction_corpora,
            "mean_best_external_f1": matrix.get("mean_best_external_f1"),
            "matrix_status": matrix.get("status"),
            "matrix_report": str(Path(config.output_dir) / "matrix" / "triple-extraction-fixture-matrix.json"),
        },
    }


def _handoff_summary(handoff: Mapping[str, Any]) -> dict[str, Any]:
    gate = _mapping(handoff.get("gate"))
    metrics = _mapping(gate.get("metrics"))
    paths = _mapping(handoff.get("paths"))
    return {
        "status": handoff.get("status"),
        "workflow_report": paths.get("workflow_report"),
        "predictions_path": paths.get("predictions"),
        "eval_report": paths.get("eval_report"),
        "artifact_manifest": paths.get("artifact_manifest"),
        "manifest_verification": paths.get("manifest_verification"),
        "gate": {
            "passed": gate.get("passed"),
            "blocking_reasons": tuple(gate.get("blocking_reasons", ())),
            "metrics": {
                "precision": metrics.get("precision"),
                "recall": metrics.get("recall"),
                "f1": metrics.get("f1"),
                "false_positive_rate": metrics.get("false_positive_rate"),
            },
        },
    }


def _matrix_summary(
    *,
    matrix: Mapping[str, Any],
    matrix_config: TripleExtractionFixtureMatrixConfig,
) -> dict[str, Any]:
    return {
        "status": matrix.get("status"),
        "report_path": str(matrix_config.summary_path),
        "artifact_manifest_path": str(matrix_config.artifact_manifest_path),
        "n_corpora": matrix.get("n_corpora"),
        "promoted_corpora": matrix.get("promoted_corpora"),
        "distinct_predicate_count": matrix.get("distinct_predicate_count"),
        "external_prediction_count": matrix.get("external_prediction_count"),
        "external_prediction_corpora": matrix.get("external_prediction_corpora"),
        "mean_best_external_f1": matrix.get("mean_best_external_f1"),
        "mean_best_f1": matrix.get("mean_best_f1"),
        "mean_f1_lift": matrix.get("mean_f1_lift"),
    }


def _write_manifest_and_optional_verification(
    *,
    config: ExternalTripleExtractorMatrixHandoffConfig,
    payload: Mapping[str, Any],
) -> dict[str, Any] | None:
    context = ArtifactVerificationContext()
    artifacts = _manifest_artifacts(config=config, payload=payload)
    manifest = context.build_artifact_manifest(
        artifacts,
        root=Path(config.artifact_manifest_path).parent,
        metadata={
            "workflow": payload.get("workflow"),
            "status": payload.get("status"),
            "gate_passed": _mapping(payload.get("gate")).get("passed"),
            "matrix_status": _mapping(payload.get("matrix")).get("status"),
            "external_prediction_count": _mapping(payload.get("matrix")).get(
                "external_prediction_count",
            ),
            "external_prediction_corpora": _mapping(payload.get("matrix")).get(
                "external_prediction_corpora",
            ),
            "mean_best_external_f1": _mapping(payload.get("matrix")).get(
                "mean_best_external_f1",
            ),
        },
        max_workers=int(config.manifest_fingerprint_workers),
    )
    _write_json(Path(config.artifact_manifest_path), manifest, compact=False)
    if config.verification_report_path is None:
        return None
    verification = context.load_and_verify_artifact_manifest(
        Path(config.artifact_manifest_path),
        recursive=True,
        max_workers=int(config.manifest_fingerprint_workers),
    ).to_dict()
    _write_json(Path(config.verification_report_path), verification, compact=config.compact_json)
    return verification


def _manifest_artifacts(
    *,
    config: ExternalTripleExtractorMatrixHandoffConfig,
    payload: Mapping[str, Any],
) -> dict[str, Path]:
    paths = _mapping(payload.get("paths"))
    artifacts = {
        "workflow_report": Path(str(paths["workflow_report"])),
        "matrix_report": Path(str(paths["matrix_report"])),
        "matrix_manifest": Path(str(paths["matrix_manifest"])),
    }
    for corpus in payload.get("corpora", ()):
        corpus_map = _mapping(corpus)
        slug = str(corpus_map.get("slug"))
        artifacts[f"corpus.{slug}.fixture_summary"] = Path(str(corpus_map["fixture_summary_path"]))
        artifacts[f"corpus.{slug}.fixture_records"] = Path(str(corpus_map["fixture_records_path"]))
        for idx, source_path in enumerate(corpus_map.get("fact_corpus_paths", ()), start=1):
            artifacts[f"corpus.{slug}.source.{idx}.{Path(str(source_path)).stem}"] = Path(
                str(source_path)
            )
        for extractor_name, handoff in _mapping(corpus_map.get("external_handoffs")).items():
            handoff_map = _mapping(handoff)
            artifacts[f"corpus.{slug}.handoff.{extractor_name}.report"] = Path(
                str(handoff_map["workflow_report"])
            )
            artifacts[f"corpus.{slug}.handoff.{extractor_name}.predictions"] = Path(
                str(handoff_map["predictions_path"])
            )
            artifacts[f"corpus.{slug}.handoff.{extractor_name}.eval_report"] = Path(
                str(handoff_map["eval_report"])
            )
            artifacts[f"corpus.{slug}.handoff.{extractor_name}.manifest"] = Path(
                str(handoff_map["artifact_manifest"])
            )
            artifacts[f"corpus.{slug}.handoff.{extractor_name}.verification"] = Path(
                str(handoff_map["manifest_verification"])
            )
    return artifacts


def _record_registry(
    *,
    config: ExternalTripleExtractorMatrixHandoffConfig,
    payload: Mapping[str, Any],
    verification: Mapping[str, Any] | None,
) -> None:
    if config.registry_path is None:
        return
    matrix = _mapping(payload.get("matrix"))
    ArtifactRegistry.load_json(Path(config.registry_path)).record_report(
        name=str(config.name),
        version=str(config.version),
        path=Path(config.workflow_report_path),
        metadata={
            "workflow": payload.get("workflow"),
            "status": payload.get("status"),
            "gate_passed": _mapping(payload.get("gate")).get("passed"),
            "artifact_manifest": str(config.artifact_manifest_path),
            "manifest_verification_report": (
                None if config.verification_report_path is None else str(config.verification_report_path)
            ),
            "manifest_verified": None if verification is None else bool(verification.get("passed")),
            "matrix_report": matrix.get("report_path"),
            "matrix_manifest": matrix.get("artifact_manifest_path"),
            "matrix_status": matrix.get("status"),
            "external_prediction_count": matrix.get("external_prediction_count"),
            "external_prediction_corpora": matrix.get("external_prediction_corpora"),
            "mean_best_external_f1": matrix.get("mean_best_external_f1"),
        },
    ).save_json()


def _config_payload(config: ExternalTripleExtractorMatrixHandoffConfig) -> dict[str, Any]:
    return {
        "corpora": [
            {
                "name": corpus.name,
                "slug": corpus.slug,
                "fact_corpus_paths": tuple(str(path) for path in corpus.fact_corpus_paths),
            }
            for corpus in config.corpora
        ],
        "extractor_names": tuple(config.extractor_commands),
        "max_facts": config.max_facts,
        "max_examples": int(config.max_examples),
        "min_augmented_f1": float(config.min_augmented_f1),
        "require_f1_lift": bool(config.require_f1_lift),
        "adversarial_negatives_per_fact": int(config.adversarial_negatives_per_fact),
        "predicate_confusions_per_fact": int(config.predicate_confusions_per_fact),
        "non_assertive_negatives_per_fact": int(config.non_assertive_negatives_per_fact),
        "ambiguity_negatives_per_fact": int(config.ambiguity_negatives_per_fact),
        "temporal_negatives_per_fact": int(config.temporal_negatives_per_fact),
        "metalinguistic_negatives_per_fact": int(config.metalinguistic_negatives_per_fact),
        "min_corpora": int(config.min_corpora),
        "min_distinct_predicates": int(config.min_distinct_predicates),
        "include_metadata": bool(config.include_metadata),
        "min_external_f1": float(config.min_external_f1),
        "min_external_precision": float(config.min_external_precision),
        "min_external_recall": float(config.min_external_recall),
        "max_external_false_positive_rate": float(config.max_external_false_positive_rate),
        "command_timeout_seconds": config.command_timeout_seconds,
    }


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


def _parse_external_command_specs(
    specs: Sequence[str],
) -> dict[str, str]:
    commands = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError("--external-extractor-command must use NAME=COMMAND format.")
        name, command = spec.split("=", 1)
        safe_name = _safe_name(name)
        command = command.strip()
        if not command:
            raise ValueError("--external-extractor-command command must be non-empty.")
        if safe_name in commands:
            raise ValueError(f"duplicate external extractor command name: {safe_name}")
        commands[safe_name] = command
    return commands


def _safe_name(value: Any) -> str:
    name = str(value).strip().casefold().replace("-", "_")
    name = "".join(char if char.isalnum() or char == "_" else "_" for char in name)
    name = "_".join(part for part in name.split("_") if part)
    if not name:
        raise ValueError("name must contain at least one alphanumeric character.")
    return name


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _float_or_none(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if numeric == numeric and numeric not in (float("inf"), float("-inf")) else None


def _validate_unit_interval(name: str, value: Any) -> None:
    numeric = float(value)
    if not (0.0 <= numeric <= 1.0):
        raise ValueError(f"{name} must be in [0, 1].")


def _validate_non_negative_int(name: str, value: Any) -> None:
    if int(value) < 0:
        raise ValueError(f"{name} must be non-negative.")


def _validate_positive_int(name: str, value: Any) -> None:
    if int(value) <= 0:
        raise ValueError(f"{name} must be positive.")


def _write_json(path: Path, payload: Mapping[str, Any], *, compact: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    indent = None if compact else 2
    path.write_text(json.dumps(payload, indent=indent, sort_keys=True) + "\n", encoding="utf-8")


def _config_from_args(args: argparse.Namespace) -> ExternalTripleExtractorMatrixHandoffConfig:
    return ExternalTripleExtractorMatrixHandoffConfig(
        corpora=_parse_corpus_specs(tuple(args.corpus)),
        extractor_commands=_parse_external_command_specs(tuple(args.external_extractor_command)),
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
        ambiguity_negatives_per_fact=args.ambiguity_negatives_per_fact,
        max_ambiguity_false_positive_rate=args.max_ambiguity_false_positive_rate,
        temporal_negatives_per_fact=args.temporal_negatives_per_fact,
        max_temporal_false_positive_rate=args.max_temporal_false_positive_rate,
        metalinguistic_negatives_per_fact=args.metalinguistic_negatives_per_fact,
        max_metalinguistic_false_positive_rate=args.max_metalinguistic_false_positive_rate,
        min_corpora=args.min_corpora,
        min_distinct_predicates=args.min_distinct_predicates,
        include_metadata=bool(args.include_metadata),
        min_external_f1=args.min_external_f1,
        min_external_precision=args.min_external_precision,
        min_external_recall=args.min_external_recall,
        max_external_false_positive_rate=args.max_external_false_positive_rate,
        command_timeout_seconds=args.command_timeout_seconds,
        manifest_fingerprint_workers=args.manifest_fingerprint_workers,
        compact_json=bool(args.compact_json),
        registry_path=args.registry,
        name=args.name,
        version=args.version,
        workflow_report_path=args.json,
        artifact_manifest_path=args.artifact_manifest,
        verification_report_path=args.verification_report,
        fail_on_blocked=bool(args.fail_on_blocked),
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run external triple extractor commands across a fixture matrix"
    )
    parser.add_argument(
        "--corpus",
        action="append",
        required=True,
        help="corpus in NAME=PATH format; repeat the same NAME to group multiple paths",
    )
    parser.add_argument(
        "--external-extractor-command",
        action="append",
        required=True,
        help="external extractor command in NAME=COMMAND format; COMMAND must use {input} and {output}",
    )
    parser.add_argument("--output-dir", default="artifacts/external-triple-extractor-matrix-handoff")
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
    parser.add_argument("--ambiguity-negatives-per-fact", type=int, default=0)
    parser.add_argument("--max-ambiguity-false-positive-rate", type=float, default=0.0)
    parser.add_argument("--temporal-negatives-per-fact", type=int, default=0)
    parser.add_argument("--max-temporal-false-positive-rate", type=float, default=0.0)
    parser.add_argument("--metalinguistic-negatives-per-fact", type=int, default=0)
    parser.add_argument("--max-metalinguistic-false-positive-rate", type=float, default=0.0)
    parser.add_argument("--min-corpora", type=int, default=2)
    parser.add_argument("--min-distinct-predicates", type=int, default=4)
    parser.add_argument("--include-metadata", action="store_true")
    parser.add_argument("--min-external-f1", type=float, default=DEFAULT_MIN_F1)
    parser.add_argument("--min-external-precision", type=float, default=DEFAULT_MIN_PRECISION)
    parser.add_argument("--min-external-recall", type=float, default=DEFAULT_MIN_RECALL)
    parser.add_argument(
        "--max-external-false-positive-rate",
        type=float,
        default=DEFAULT_MAX_FALSE_POSITIVE_RATE,
    )
    parser.add_argument("--command-timeout-seconds", type=float, default=None)
    parser.add_argument("--manifest-fingerprint-workers", type=int, default=1)
    parser.add_argument("--compact-json", action="store_true")
    parser.add_argument("--json", default=None)
    parser.add_argument("--artifact-manifest", default=None)
    parser.add_argument("--verification-report", default=None)
    parser.add_argument("--registry", default=None)
    parser.add_argument("--name", default=DEFAULT_WORKFLOW_NAME)
    parser.add_argument("--version", default="0.1")
    parser.add_argument("--fail-on-blocked", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    config = _config_from_args(build_arg_parser().parse_args(argv))
    run_external_triple_extractor_matrix_handoff(config)


if __name__ == "__main__":
    main()
