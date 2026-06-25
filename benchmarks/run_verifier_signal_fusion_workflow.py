"""Run verifier-signal fusion from local evidence artifacts.

This workflow turns non-oracle local retrieval/self-check verifier outputs into
standard score-dump columns, then evaluates geometry-by-uncertainty fusion over
those columns. It does not load models or call the network.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, MutableMapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.build_evidence_fixture import (  # noqa: E402
    RETRIEVER_BACKENDS,
    build_evidence_fixture,
    build_evidence_input_provenance,
    load_corpus,
)
from benchmarks.build_selfcheck_fixture import build_selfcheck_fixture, load_sample_payloads  # noqa: E402
from benchmarks.build_verifier_signal_score_dump import DEFAULT_VERIFIER_SIGNALS  # noqa: E402
from benchmarks.build_verifier_signal_score_dump import (  # noqa: E402
    build_report as build_verifier_signal_score_dump_report,
)
from benchmarks.eval_score_ensemble import (  # noqa: E402
    GEOMETRY_FUSION_METHODS,
    METHODS,
    build_ensemble_report,
    build_geometry_fusion_artifact_from_score_dump,
)
from benchmarks.eval_verifier_ensemble import build_verifier_ensemble_report  # noqa: E402
from eigentruth.eval.score_dump import load_score_dump  # noqa: E402
from eigentruth.registry import ArtifactVerificationContext, build_artifact_manifest  # noqa: E402

DEFAULT_FUSION_SIGNALS = (
    "truth_proj",
    "subspace_resid",
    "eigenscore",
    "verifier_refuted",
    "verifier_not_supported",
    "verifier_refute_confidence",
    "verifier_uncertainty",
    "selfcheck_refute_rate",
)
DEFAULT_GEOMETRY_SIGNALS = ("truth_proj", "subspace_resid", "eigenscore")
DEFAULT_UNCERTAINTY_SIGNALS = (
    "verifier_refuted",
    "verifier_refute_confidence",
    "verifier_not_supported",
)


@dataclass(frozen=True)
class VerifierSignalFusionWorkflowConfig:
    """Configuration for the verifier-signal fusion workflow."""

    score_dumps: Sequence[tuple[str, Path]]
    output_dir: Path
    claims_path: Path | None = None
    corpus_paths: Sequence[Path] = ()
    sample_paths: Sequence[Path] = ()
    qa_corpus_path: Path | None = None
    state_path: Path | None = None
    signal: str = "truth_proj"
    direction: str | None = None
    alphas: Sequence[float] = (0.10,)
    repeats: int = 20
    seed: int = 0
    best_alpha: float = 0.10
    keep_signals: Sequence[str] | None = None
    verifier_signals: Sequence[str] = DEFAULT_VERIFIER_SIGNALS
    fusion_signals: Sequence[str] = DEFAULT_FUSION_SIGNALS
    methods: Sequence[str] = METHODS
    geometry_signals: Sequence[str] = DEFAULT_GEOMETRY_SIGNALS
    uncertainty_signals: Sequence[str] = DEFAULT_UNCERTAINTY_SIGNALS
    geometry_method: str = "mean_rank"
    uncertainty_method: str = "mean_rank"
    geometry_fusion_methods: Sequence[str] = GEOMETRY_FUSION_METHODS
    query_field: str = "answer"
    retriever_backend: str = "memory"
    retriever_index_path: Path | None = None
    include_label_metadata: bool = False
    verifier_min_overlap: float = 0.65
    retriever_min_overlap: float = 0.20
    retrieval_limit: int = 5
    selfcheck_min_samples: int = 2
    selfcheck_min_overlap: float = 0.65
    selfcheck_support_threshold: float = 0.60
    selfcheck_refute_threshold: float = 0.50
    selfcheck_early_stop: bool = False
    selfcheck_max_samples: int | None = None
    staged_verification: bool = False
    staged_alpha: float = 0.10
    compact_json: bool = False
    verify_manifest: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "score_dumps",
            tuple((str(name), Path(path)) for name, path in self.score_dumps),
        )
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        for attr in ("claims_path", "qa_corpus_path", "state_path", "retriever_index_path"):
            value = getattr(self, attr)
            if value is not None:
                object.__setattr__(self, attr, Path(value))
        object.__setattr__(self, "corpus_paths", tuple(Path(path) for path in self.corpus_paths))
        object.__setattr__(self, "sample_paths", tuple(Path(path) for path in self.sample_paths))
        for attr in (
            "alphas",
            "verifier_signals",
            "fusion_signals",
            "methods",
            "geometry_signals",
            "uncertainty_signals",
            "geometry_fusion_methods",
        ):
            object.__setattr__(self, attr, tuple(getattr(self, attr)))
        if self.keep_signals is not None:
            object.__setattr__(self, "keep_signals", tuple(self.keep_signals))
        if not self.score_dumps:
            raise ValueError("score_dumps must contain at least one score dump.")
        if len({name for name, _ in self.score_dumps}) != len(self.score_dumps):
            raise ValueError("score dump names must be unique.")
        if int(self.repeats) < 1:
            raise ValueError("repeats must be >= 1.")
        if any(not (0.0 < float(alpha) < 1.0) for alpha in self.alphas):
            raise ValueError("alphas must be in (0, 1).")
        if not (0.0 < float(self.best_alpha) < 1.0):
            raise ValueError("best_alpha must be in (0, 1).")
        if not self.fusion_signals:
            raise ValueError("fusion_signals must contain at least one signal.")
        if bool(self.geometry_signals) != bool(self.uncertainty_signals):
            raise ValueError("geometry_signals and uncertainty_signals must be provided together.")
        if self.retriever_backend not in RETRIEVER_BACKENDS:
            raise ValueError(f"retriever_backend must be one of: {', '.join(RETRIEVER_BACKENDS)}.")
        if self.retriever_backend == "memory" and self.retriever_index_path is not None:
            raise ValueError("retriever_index_path is only supported with sqlite_fts or auto backends.")
        if self.claims_path is not None and (self.corpus_paths or self.sample_paths):
            raise ValueError("claims_path cannot be combined with corpus_paths or sample_paths.")

    @property
    def resolved_claims_path(self) -> Path | None:
        if self.claims_path is not None:
            return self.claims_path
        if not self.corpus_paths and not self.sample_paths:
            return None
        return self.output_dir / "verifier-claims.json"

    @property
    def verifier_report_path(self) -> Path:
        return self.output_dir / "verifier-ensemble-report.json"

    @property
    def verified_records_path(self) -> Path:
        return self.output_dir / "verified-records.jsonl"

    @property
    def score_ensemble_report_path(self) -> Path:
        return self.output_dir / "score-ensemble-report.json"

    @property
    def artifact_manifest_path(self) -> Path:
        return self.output_dir / "artifact-manifest.json"

    @property
    def manifest_verification_path(self) -> Path:
        return self.output_dir / "manifest-verification.json"

    @property
    def workflow_report_path(self) -> Path:
        return self.output_dir / "verifier-signal-fusion-workflow.json"


def run_verifier_signal_fusion_workflow(
    config: VerifierSignalFusionWorkflowConfig,
) -> dict[str, Any]:
    """Run verifier-signal score construction and score fusion."""
    started = time.perf_counter()
    profile: dict[str, float] = {}
    config.output_dir.mkdir(parents=True, exist_ok=True)

    with _profile_phase(profile, "build_claims_fixture"):
        claims_fixture = _build_or_load_claims_fixture(config)

    with _profile_phase(profile, "build_verifier_report"):
        verifier_report = build_verifier_ensemble_report(
            config.score_dumps,
            signal=config.signal,
            claims_path=config.resolved_claims_path,
            qa_corpus_path=config.qa_corpus_path,
            state_path=config.state_path,
            direction=config.direction,
            alphas=tuple(float(alpha) for alpha in config.alphas),
            repeats=int(config.repeats),
            seed=int(config.seed),
            verifier_min_overlap=float(config.verifier_min_overlap),
            retriever_min_overlap=float(config.retriever_min_overlap),
            retrieval_limit=int(config.retrieval_limit),
            selfcheck_min_samples=int(config.selfcheck_min_samples),
            selfcheck_min_overlap=float(config.selfcheck_min_overlap),
            selfcheck_support_threshold=float(config.selfcheck_support_threshold),
            selfcheck_refute_threshold=float(config.selfcheck_refute_threshold),
            selfcheck_early_stop=bool(config.selfcheck_early_stop),
            selfcheck_max_samples=config.selfcheck_max_samples,
            staged_verification=bool(config.staged_verification),
            staged_alpha=float(config.staged_alpha),
            verified_records_path=config.verified_records_path,
        )
    with _profile_phase(profile, "write_verifier_report"):
        _write_json(config.verifier_report_path, verifier_report, compact=config.compact_json)

    enhanced_score_dumps: list[tuple[str, Path]] = []
    enhanced_reports: dict[str, str] = {}
    with _profile_phase(profile, "build_verifier_signal_score_dumps"):
        for run_name, source_path in config.score_dumps:
            output_path = config.output_dir / f"{run_name}-enhanced-scores.manifest.json"
            report_path = config.output_dir / f"{run_name}-enhanced-score-report.json"
            report = build_verifier_signal_score_dump_report(
                input_scores=source_path,
                verified_records_jsonl=config.verified_records_path,
                output=output_path,
                output_format="jsonl",
                run_name=run_name,
                keep_signals=config.keep_signals,
                verifier_signals=config.verifier_signals,
            )
            _write_json(report_path, report, compact=config.compact_json)
            enhanced_score_dumps.append((run_name, output_path))
            enhanced_reports[run_name] = str(report_path)

    with _profile_phase(profile, "build_score_ensemble_report"):
        score_ensemble_report = build_ensemble_report(
            enhanced_score_dumps,
            signals=config.fusion_signals,
            methods=config.methods,
            geometry_signals=config.geometry_signals,
            uncertainty_signals=config.uncertainty_signals,
            geometry_method=config.geometry_method,
            uncertainty_method=config.uncertainty_method,
            geometry_fusion_methods=config.geometry_fusion_methods,
            alphas=tuple(float(alpha) for alpha in config.alphas),
            repeats=int(config.repeats),
            seed=int(config.seed),
            best_alpha=float(config.best_alpha),
        )
    with _profile_phase(profile, "write_score_ensemble_report"):
        _write_json(config.score_ensemble_report_path, score_ensemble_report, compact=config.compact_json)

    geometry_artifacts: dict[str, str] = {}
    if config.geometry_signals and config.uncertainty_signals:
        with _profile_phase(profile, "build_geometry_fusion_artifacts"):
            for run_name, enhanced_path in enhanced_score_dumps:
                run_payload = _run_payload_by_name(score_ensemble_report, run_name)
                best_geometry = run_payload.get("best_geometry_fusion_at_alpha")
                if best_geometry is None:
                    continue
                artifact_path = config.output_dir / f"{run_name}-geometry-fusion-artifact.json"
                artifact = build_geometry_fusion_artifact_from_score_dump(
                    (run_name, enhanced_path),
                    geometry_signals=config.geometry_signals,
                    uncertainty_signals=config.uncertainty_signals,
                    geometry_method=config.geometry_method,
                    uncertainty_method=config.uncertainty_method,
                    fusion_method=str(best_geometry["name"]),
                    alpha=float(config.best_alpha),
                    cache={},
                )
                artifact.save_json(artifact_path)
                geometry_artifacts[run_name] = str(artifact_path)

    manifest_metadata = _manifest_metadata(
        config,
        claims_fixture=claims_fixture,
        verifier_report=verifier_report,
        score_ensemble_report=score_ensemble_report,
        geometry_artifacts=geometry_artifacts,
        profile=profile,
        total_seconds=time.perf_counter() - started,
    )
    with _profile_phase(profile, "write_artifact_manifest"):
        manifest = _write_artifact_manifest(
            config,
            enhanced_score_dumps=enhanced_score_dumps,
            enhanced_reports=enhanced_reports,
            geometry_artifacts=geometry_artifacts,
            metadata=manifest_metadata,
        )

    manifest_verification = None
    if config.verify_manifest:
        with _profile_phase(profile, "verify_artifact_manifest"):
            context = ArtifactVerificationContext()
            manifest_verification = context.load_and_verify_artifact_manifest(
                config.artifact_manifest_path,
                root=config.artifact_manifest_path.parent,
            ).to_dict()
            _write_json(config.manifest_verification_path, manifest_verification, compact=False)

    profile["total_seconds"] = time.perf_counter() - started
    payload = {
        "schema_version": 1,
        "workflow": "verifier_signal_fusion_workflow",
        "config": _config_payload(config),
        "claims_path": None if config.resolved_claims_path is None else str(config.resolved_claims_path),
        "verifier_report_path": str(config.verifier_report_path),
        "verified_records_path": str(config.verified_records_path),
        "enhanced_score_dumps": {name: str(path) for name, path in enhanced_score_dumps},
        "enhanced_score_reports": enhanced_reports,
        "score_ensemble_report_path": str(config.score_ensemble_report_path),
        "geometry_fusion_artifacts": geometry_artifacts,
        "artifact_manifest_path": str(config.artifact_manifest_path),
        "manifest_verification_path": (
            None if manifest_verification is None else str(config.manifest_verification_path)
        ),
        "claims_summary": None if claims_fixture is None else claims_fixture.get("summary"),
        "verifier_summary": _verifier_summary(verifier_report),
        "fusion_summary": _fusion_summary(score_ensemble_report),
        "manifest_summary": manifest.get("summary"),
        "manifest_verification": manifest_verification,
        "profile": dict(profile),
    }
    _write_json(config.workflow_report_path, payload, compact=config.compact_json)
    return payload


def _build_or_load_claims_fixture(
    config: VerifierSignalFusionWorkflowConfig,
) -> dict[str, Any] | None:
    claims_path = config.resolved_claims_path
    if config.claims_path is not None:
        payload = json.loads(config.claims_path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("claims_path must contain a JSON object.")
        return dict(payload)
    if claims_path is None:
        return None

    first_scores_path = config.score_dumps[0][1]
    score_dump = load_score_dump(
        first_scores_path,
        allow_missing_scores=True,
        require_statements=True,
    ).to_mapping()
    _validate_generated_claims_alignment(config, score_dump)
    retrieval_fixture = None
    if config.corpus_paths:
        corpus = load_corpus(config.corpus_paths)
        retrieval_fixture = build_evidence_fixture(
            score_dump,
            corpus,
            retriever_min_overlap=float(config.retriever_min_overlap),
            retrieval_limit=int(config.retrieval_limit),
            query_field=config.query_field,
            retriever_backend=config.retriever_backend,
            retriever_index_path=config.retriever_index_path,
            include_label_metadata=bool(config.include_label_metadata),
        )
        retrieval_fixture["input_provenance"] = build_evidence_input_provenance(
            scores_path=first_scores_path,
            corpus_paths=config.corpus_paths,
            score_dump=score_dump,
            retriever_backend=config.retriever_backend,
            retriever_index_path=config.retriever_index_path,
            retriever_min_overlap=float(config.retriever_min_overlap),
            retrieval_limit=int(config.retrieval_limit),
            query_field=config.query_field,
            include_label_metadata=bool(config.include_label_metadata),
        )

    selfcheck_fixture = None
    if config.sample_paths:
        sample_payloads = load_sample_payloads(config.sample_paths)
        selfcheck_fixture = build_selfcheck_fixture(
            score_dump,
            sample_payloads,
            min_samples=int(config.selfcheck_min_samples),
            include_empty_records=True,
        )
        if not config.include_label_metadata:
            _strip_score_labels(selfcheck_fixture)

    claims_fixture = _merge_claim_fixtures(
        retrieval_fixture,
        selfcheck_fixture,
        include_label_metadata=bool(config.include_label_metadata),
    )
    if selfcheck_fixture is not None:
        provenance = claims_fixture.setdefault("input_provenance", {})
        if isinstance(provenance, MutableMapping):
            selfcheck_provenance = provenance.setdefault("selfcheck", {})
            if isinstance(selfcheck_provenance, MutableMapping):
                selfcheck_provenance["sample_paths"] = [str(path) for path in config.sample_paths]
    _write_json(claims_path, claims_fixture, compact=config.compact_json)
    return claims_fixture


def _validate_generated_claims_alignment(
    config: VerifierSignalFusionWorkflowConfig,
    first_score_dump: Mapping[str, Any],
) -> None:
    """Ensure generated claims can be reused across all requested score dumps."""
    expected_statements = tuple(_statement_identity(item) for item in first_score_dump.get("statements", ()))
    expected_labels = tuple(int(label) for label in first_score_dump.get("labels", ()))
    for run_name, path in config.score_dumps[1:]:
        dump = load_score_dump(
            path,
            allow_missing_scores=True,
            require_statements=True,
        ).to_mapping()
        labels = tuple(int(label) for label in dump.get("labels", ()))
        statements = tuple(_statement_identity(item) for item in dump.get("statements", ()))
        if labels != expected_labels or statements != expected_statements:
            raise ValueError(
                "generated retrieval/selfcheck claims require aligned labels and statements "
                f"across score dumps; run {run_name!r} at {path} differs from {config.score_dumps[0][0]!r}."
            )


def _statement_identity(value: Any) -> tuple[str, str, str, str]:
    if not isinstance(value, Mapping):
        raise ValueError("score dump statements must be JSON objects.")
    return (
        str(value.get("claim_id", "")),
        str(value.get("question", "")),
        str(value.get("answer", "")),
        str(value.get("text", value.get("claim", ""))),
    )


def _merge_claim_fixtures(
    retrieval_fixture: Mapping[str, Any] | None,
    selfcheck_fixture: Mapping[str, Any] | None,
    *,
    include_label_metadata: bool,
) -> dict[str, Any]:
    if retrieval_fixture is None and selfcheck_fixture is None:
        raise ValueError("at least one generated fixture is required.")
    if retrieval_fixture is None:
        merged = dict(selfcheck_fixture or {})
        merged["fixture_type"] = "selfcheck_samples"
        return merged
    if selfcheck_fixture is None:
        return dict(retrieval_fixture)

    retrieval_records = tuple(_record_copy(item) for item in retrieval_fixture.get("records", ()))
    selfcheck_records = tuple(_record_copy(item) for item in selfcheck_fixture.get("records", ()))
    if len(retrieval_records) != len(selfcheck_records):
        raise ValueError("retrieval and selfcheck fixtures must have the same record count.")

    records = []
    for idx, (retrieval_record, selfcheck_record) in enumerate(zip(retrieval_records, selfcheck_records)):
        if str(retrieval_record.get("claim_id", "")) != str(selfcheck_record.get("claim_id", "")):
            raise ValueError(f"fixture claim_id mismatch at index {idx}.")
        metadata = {
            **dict(retrieval_record.get("metadata", {})),
            "selfcheck": dict(selfcheck_record.get("metadata", {}).get("selfcheck", {})),
        }
        if not include_label_metadata:
            metadata.pop("score_label", None)
        records.append({
            **retrieval_record,
            "claim_metadata": {
                **dict(retrieval_record.get("claim_metadata", {})),
                **dict(selfcheck_record.get("claim_metadata", {})),
            },
            "selfcheck_samples": list(selfcheck_record.get("selfcheck_samples", ())),
            "metadata": metadata,
        })

    retrieval_summary = dict(retrieval_fixture.get("summary", {}))
    selfcheck_summary = dict(selfcheck_fixture.get("summary", {}))
    return {
        "schema_version": 1,
        "fixture_type": "local_retrieval_selfcheck_evidence",
        "description": (
            "Combined local retrieval and self-consistency fixture. Labels are not used "
            "to retrieve or verify claims."
        ),
        "label_usage": {
            "labels_used_for_retrieval": False,
            "labels_used_for_selfcheck": False,
            "labels_copied_to_record_metadata": bool(include_label_metadata),
        },
        "retriever": dict(retrieval_fixture.get("retriever", {})),
        "selfcheck": dict(selfcheck_fixture.get("selfcheck", {})),
        "summary": {
            "n_records": len(records),
            "records_with_hits": sum(1 for record in records if record.get("retrieval_documents")),
            "records_with_samples": sum(1 for record in records if record.get("selfcheck_samples")),
            "records_meeting_min_samples": selfcheck_summary.get("records_meeting_min_samples", 0),
            "total_hits": retrieval_summary.get("total_hits", 0),
            "total_samples": selfcheck_summary.get("total_samples", 0),
        },
        "input_provenance": {
            "retrieval": retrieval_fixture.get("input_provenance"),
            "selfcheck": {
                "builder": "build_selfcheck_fixture",
                "sample_paths": [],
            },
        },
        "records": records,
    }


def _strip_score_labels(fixture: MutableMapping[str, Any]) -> None:
    label_usage = fixture.get("label_usage")
    if isinstance(label_usage, MutableMapping):
        label_usage["labels_copied_to_record_metadata"] = False
    for record in fixture.get("records", ()):
        if isinstance(record, MutableMapping):
            metadata = record.get("metadata")
            if isinstance(metadata, MutableMapping):
                metadata.pop("score_label", None)


def _record_copy(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("fixture records must be JSON objects.")
    return json.loads(json.dumps(value))


def _write_artifact_manifest(
    config: VerifierSignalFusionWorkflowConfig,
    *,
    enhanced_score_dumps: Sequence[tuple[str, Path]],
    enhanced_reports: Mapping[str, str],
    geometry_artifacts: Mapping[str, str],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    artifacts: dict[str, str | Path | None] = {
        "verifier_report": config.verifier_report_path,
        "verified_records": config.verified_records_path,
        "score_ensemble_report": config.score_ensemble_report_path,
    }
    if config.resolved_claims_path is not None:
        artifacts["claims"] = config.resolved_claims_path
    if config.qa_corpus_path is not None:
        artifacts["qa_corpus"] = config.qa_corpus_path
    if config.state_path is not None:
        artifacts["state_source"] = config.state_path
    for idx, path in enumerate(config.corpus_paths, start=1):
        artifacts[f"retrieval_corpora.{idx}.{path.stem}"] = path
    for idx, path in enumerate(config.sample_paths, start=1):
        artifacts[f"selfcheck_samples.{idx}.{path.stem}"] = path
    for run_name, path in config.score_dumps:
        artifacts[f"source_scores.{run_name}"] = path
    for run_name, path in enhanced_score_dumps:
        artifacts[f"enhanced_scores.{run_name}"] = path
        artifacts[f"enhanced_records.{run_name}"] = path.with_suffix(".records.jsonl")
    for run_name, path in enhanced_reports.items():
        artifacts[f"enhanced_report.{run_name}"] = path
    for run_name, path in geometry_artifacts.items():
        artifacts[f"geometry_fusion_artifact.{run_name}"] = path

    manifest = build_artifact_manifest(
        artifacts,
        root=config.artifact_manifest_path.parent,
        metadata=metadata,
    )
    _write_json(config.artifact_manifest_path, manifest, compact=False)
    return manifest


def _manifest_metadata(
    config: VerifierSignalFusionWorkflowConfig,
    *,
    claims_fixture: Mapping[str, Any] | None,
    verifier_report: Mapping[str, Any],
    score_ensemble_report: Mapping[str, Any],
    geometry_artifacts: Mapping[str, str],
    profile: Mapping[str, float],
    total_seconds: float,
) -> dict[str, Any]:
    fusion_summary = _fusion_summary(score_ensemble_report)
    verifier_summary = _verifier_summary(verifier_report)
    claims_summary = None if claims_fixture is None else dict(claims_fixture.get("summary", {}))
    return {
        "runner": "run_verifier_signal_fusion_workflow",
        "workflow": "verifier_signal_fusion_workflow",
        "score_names": [name for name, _ in config.score_dumps],
        "signal": config.signal,
        "fusion_signals": list(config.fusion_signals),
        "geometry_signals": list(config.geometry_signals),
        "uncertainty_signals": list(config.uncertainty_signals),
        "best_alpha": float(config.best_alpha),
        "claims_fixture_type": None if claims_fixture is None else claims_fixture.get("fixture_type"),
        "claims_summary": claims_summary,
        "verifier_summary": verifier_summary,
        "fusion_summary": fusion_summary,
        "geometry_artifact_count": len(geometry_artifacts),
        "uses_non_oracle_local_corpus": bool(config.corpus_paths),
        "uses_selfcheck_samples": bool(config.sample_paths),
        "labels_used_for_retrieval": (
            None
            if claims_fixture is None
            else dict(claims_fixture.get("label_usage", {})).get("labels_used_for_retrieval")
        ),
        "labels_copied_to_record_metadata": (
            None
            if claims_fixture is None
            else dict(claims_fixture.get("label_usage", {})).get("labels_copied_to_record_metadata")
        ),
        "profile": dict(profile),
        "total_seconds": float(total_seconds),
    }


def _verifier_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    runs = []
    for run in report.get("runs", ()):
        if not isinstance(run, Mapping):
            continue
        quality = dict(run.get("verification_quality", {}))
        route_summary = dict(run.get("route_summary", {}))
        best_alpha_payload = dict(run.get("alphas", {}).get(str(float(report.get("best_alpha", 0.1))), {}))
        verified = dict(best_alpha_payload.get("verified", {}))
        runs.append({
            "name": run.get("name"),
            "true_supported_rate": quality.get("true_supported_rate"),
            "false_refuted_rate": quality.get("false_refuted_rate"),
            "selected_counts": route_summary.get("selected_counts"),
            "verified_false_alarm": verified.get("false_alarm"),
            "verified_detection": verified.get("detection"),
        })
    return {"runs": runs}


def _fusion_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    runs = []
    for run in report.get("runs", ()):
        if not isinstance(run, Mapping):
            continue
        runs.append({
            "name": run.get("name"),
            "best_single_at_alpha": run.get("best_single_at_alpha"),
            "best_ensemble_at_alpha": run.get("best_ensemble_at_alpha"),
            "best_geometry_fusion_at_alpha": run.get("best_geometry_fusion_at_alpha"),
        })
    return {"runs": runs}


def _run_payload_by_name(report: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    for run in report.get("runs", ()):
        if isinstance(run, Mapping) and str(run.get("name")) == str(name):
            return run
    raise ValueError(f"score ensemble report has no run named {name!r}.")


def _config_payload(config: VerifierSignalFusionWorkflowConfig) -> dict[str, Any]:
    return {
        "score_dumps": {name: str(path) for name, path in config.score_dumps},
        "claims_path": None if config.claims_path is None else str(config.claims_path),
        "corpus_paths": [str(path) for path in config.corpus_paths],
        "sample_paths": [str(path) for path in config.sample_paths],
        "qa_corpus_path": None if config.qa_corpus_path is None else str(config.qa_corpus_path),
        "state_path": None if config.state_path is None else str(config.state_path),
        "signal": config.signal,
        "direction": config.direction,
        "alphas": [float(alpha) for alpha in config.alphas],
        "repeats": int(config.repeats),
        "seed": int(config.seed),
        "best_alpha": float(config.best_alpha),
        "keep_signals": None if config.keep_signals is None else list(config.keep_signals),
        "verifier_signals": list(config.verifier_signals),
        "fusion_signals": list(config.fusion_signals),
        "methods": list(config.methods),
        "geometry_signals": list(config.geometry_signals),
        "uncertainty_signals": list(config.uncertainty_signals),
        "geometry_method": config.geometry_method,
        "uncertainty_method": config.uncertainty_method,
        "geometry_fusion_methods": list(config.geometry_fusion_methods),
        "query_field": config.query_field,
        "retriever_backend": config.retriever_backend,
        "retriever_index_path": None if config.retriever_index_path is None else str(config.retriever_index_path),
        "include_label_metadata": bool(config.include_label_metadata),
        "verifier_min_overlap": float(config.verifier_min_overlap),
        "retriever_min_overlap": float(config.retriever_min_overlap),
        "retrieval_limit": int(config.retrieval_limit),
        "selfcheck_min_samples": int(config.selfcheck_min_samples),
        "selfcheck_min_overlap": float(config.selfcheck_min_overlap),
        "selfcheck_support_threshold": float(config.selfcheck_support_threshold),
        "selfcheck_refute_threshold": float(config.selfcheck_refute_threshold),
        "selfcheck_early_stop": bool(config.selfcheck_early_stop),
        "selfcheck_max_samples": config.selfcheck_max_samples,
        "staged_verification": bool(config.staged_verification),
        "staged_alpha": float(config.staged_alpha),
        "verify_manifest": bool(config.verify_manifest),
    }


@contextmanager
def _profile_phase(profile: MutableMapping[str, float], name: str) -> Iterator[None]:
    started = time.perf_counter()
    try:
        yield
    finally:
        profile[name] = profile.get(name, 0.0) + (time.perf_counter() - started)


def _write_json(path: Path, payload: Mapping[str, Any], *, compact: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if compact:
        text = json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    else:
        text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path.write_text(text, encoding="utf-8")


def _parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        path = Path(value)
        return path.stem, path
    name, path = value.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError("scores name cannot be empty.")
    return name, Path(path)


def _parse_csv(value: str | None, *, name: str) -> tuple[str, ...] | None:
    if value is None:
        return None
    parts = tuple(part.strip() for part in value.split(",") if part.strip())
    if not parts:
        raise ValueError(f"{name} must contain at least one value.")
    if len(set(parts)) != len(parts):
        raise ValueError(f"{name} must contain unique values.")
    return parts


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run from CLI-style arguments."""
    config = VerifierSignalFusionWorkflowConfig(
        score_dumps=tuple(_parse_named_path(value) for value in args.scores),
        output_dir=Path(args.output_dir),
        claims_path=None if args.claims is None else Path(args.claims),
        corpus_paths=tuple(Path(path) for path in args.corpus or ()),
        sample_paths=tuple(Path(path) for path in args.samples or ()),
        qa_corpus_path=None if args.qa_corpus is None else Path(args.qa_corpus),
        state_path=None if args.state_source is None else Path(args.state_source),
        signal=args.signal,
        direction=args.direction,
        alphas=tuple(float(value) for value in (_parse_csv(args.alphas, name="alphas") or ())),
        repeats=args.repeats,
        seed=args.seed,
        best_alpha=args.best_alpha,
        keep_signals=_parse_csv(args.keep_signals, name="keep_signals"),
        verifier_signals=_parse_csv(args.verifier_signals, name="verifier_signals") or DEFAULT_VERIFIER_SIGNALS,
        fusion_signals=_parse_csv(args.fusion_signals, name="fusion_signals") or DEFAULT_FUSION_SIGNALS,
        methods=_parse_csv(args.methods, name="methods") or METHODS,
        geometry_signals=_parse_csv(args.geometry_signals, name="geometry_signals") or DEFAULT_GEOMETRY_SIGNALS,
        uncertainty_signals=(
            _parse_csv(args.uncertainty_signals, name="uncertainty_signals")
            or DEFAULT_UNCERTAINTY_SIGNALS
        ),
        geometry_method=args.geometry_method,
        uncertainty_method=args.uncertainty_method,
        geometry_fusion_methods=(
            _parse_csv(args.geometry_fusion_methods, name="geometry_fusion_methods")
            or GEOMETRY_FUSION_METHODS
        ),
        query_field=args.query_field,
        retriever_backend=args.retriever_backend,
        retriever_index_path=None if args.retriever_index_path is None else Path(args.retriever_index_path),
        include_label_metadata=not bool(args.omit_label_metadata),
        verifier_min_overlap=args.verifier_min_overlap,
        retriever_min_overlap=args.retriever_min_overlap,
        retrieval_limit=args.retrieval_limit,
        selfcheck_min_samples=args.selfcheck_min_samples,
        selfcheck_min_overlap=args.selfcheck_min_overlap,
        selfcheck_support_threshold=args.selfcheck_support_threshold,
        selfcheck_refute_threshold=args.selfcheck_refute_threshold,
        selfcheck_early_stop=args.selfcheck_early_stop,
        selfcheck_max_samples=args.selfcheck_max_samples,
        staged_verification=args.staged_verification,
        staged_alpha=args.staged_alpha,
        compact_json=args.compact_json,
        verify_manifest=not bool(args.no_verify_manifest),
    )
    payload = run_verifier_signal_fusion_workflow(config)
    print(
        f"verifier_signal_fusion_workflow_ok runs={len(config.score_dumps)} "
        f"manifest={payload['artifact_manifest_path']}"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run verifier-signal fusion over local evidence artifacts")
    parser.add_argument("--scores", action="append", required=True,
                        help="score dump path, optionally named as name=path; repeatable")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--claims", default=None, help="optional prebuilt claim fixture JSON")
    parser.add_argument("--corpus", action="append", default=None,
                        help="local retrieval corpus path; repeatable")
    parser.add_argument("--samples", action="append", default=None,
                        help="sampled generation JSON/JSONL for selfcheck; repeatable")
    parser.add_argument("--qa-corpus", default=None)
    parser.add_argument("--state-source", default=None)
    parser.add_argument("--signal", default="truth_proj")
    parser.add_argument("--direction", choices=("higher", "lower"), default=None)
    parser.add_argument("--alphas", default="0.1")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--best-alpha", type=float, default=0.10)
    parser.add_argument("--keep-signals", default=None)
    parser.add_argument("--verifier-signals", default=",".join(DEFAULT_VERIFIER_SIGNALS))
    parser.add_argument("--fusion-signals", default=",".join(DEFAULT_FUSION_SIGNALS))
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--geometry-signals", default=",".join(DEFAULT_GEOMETRY_SIGNALS))
    parser.add_argument("--uncertainty-signals", default=",".join(DEFAULT_UNCERTAINTY_SIGNALS))
    parser.add_argument("--geometry-method", default="mean_rank")
    parser.add_argument("--uncertainty-method", default="mean_rank")
    parser.add_argument("--geometry-fusion-methods", default=",".join(GEOMETRY_FUSION_METHODS))
    parser.add_argument("--query-field", choices=("text", "answer", "question", "question_answer"), default="answer")
    parser.add_argument("--retriever-backend", choices=RETRIEVER_BACKENDS, default="memory")
    parser.add_argument("--retriever-index-path", default=None)
    parser.add_argument("--omit-label-metadata", action="store_true", default=True)
    parser.add_argument("--include-label-metadata", dest="omit_label_metadata", action="store_false")
    parser.add_argument("--verifier-min-overlap", type=float, default=0.65)
    parser.add_argument("--retriever-min-overlap", type=float, default=0.20)
    parser.add_argument("--retrieval-limit", type=int, default=5)
    parser.add_argument("--selfcheck-min-samples", type=int, default=2)
    parser.add_argument("--selfcheck-min-overlap", type=float, default=0.65)
    parser.add_argument("--selfcheck-support-threshold", type=float, default=0.60)
    parser.add_argument("--selfcheck-refute-threshold", type=float, default=0.50)
    parser.add_argument("--selfcheck-early-stop", action="store_true")
    parser.add_argument("--selfcheck-max-samples", type=int, default=None)
    parser.add_argument("--staged-verification", action="store_true")
    parser.add_argument("--staged-alpha", type=float, default=0.10)
    parser.add_argument("--compact-json", action="store_true")
    parser.add_argument("--no-verify-manifest", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
